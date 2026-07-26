unit MainForm;

{ 제스처 키오스크 데모 UI (델파이7) - 파이프 수신 + 포커스 이동 (2026-07-23).

  회사 키오스크 프로그램에 들어갈 "포커스 이동" 기능의 동작 증명용 데모:
  - 엔진(python)을 자식 프로세스로 실행하고 stdout 파이프로 이벤트 수신
  - 이벤트 7종 처리: left/right/top/bottom = 포커스 이동, ok = 실행,
    back = 이전 화면, home = 처음 화면 (docs/델파이7_연동가이드.md §3)
  - 화면 3종: 메뉴(증명서 4종 2x2) -> 발급 확인 -> 완료
  - 첫 화면에서 back/home은 무시(로그만) - 화면 이탈 사고 방지 정책
  - 키보드 폴백: 방향키/Enter/Backspace/Home - 엔진 없이도 포커스 동작 확인 가능

  실전 이식 시: HandleEvent 이하의 포커스·화면 로직을 실제 발급 화면의
  컨트롤 목록에 연결하면 된다 (수신부 EngineProcess.pas는 그대로 재사용). }

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms,
  StdCtrls, ExtCtrls, ComCtrls, EngineProcess;

const
  // ─── 엔진 실행 설정 - 배포 PC에 맞게 이 두 줄만 조정한다 ───────────────
  //  exe가 gesture_kiosk\delphi_ui\ 안에서 실행된다는 가정 (WorkDir = 상위 폴더).
  //  win 브랜치가 아니면(개발 시험) 'cmd /c python scripts\run_demo.py' 로 교체.
  ENGINE_COMMAND = 'cmd /c run.bat --headless';
  ENGINE_WORKDIR = '..';

type
  TScreenKind = (skMenu, skConfirm, skDone);

  TForm1 = class(TForm)
    LogMemo: TMemo;
    StatusBar: TStatusBar;
    RestartTimer: TTimer;
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
    procedure RestartTimerTimer(Sender: TObject);
  private
    FEngine: TEngineProcess;
    FScreenKind: TScreenKind;
    FTitleLabel: TLabel;
    FItems: TList;          // 현재 화면의 포커스 대상(TPanel) 목록 - 격자 순서
    FCols: Integer;         // 격자 열 수 (포커스 상하 이동 = ±FCols)
    FFocusIndex: Integer;
    FSelectedDoc: string;
    FEngineAlive: Boolean;
    procedure StartEngine;
    procedure BuildScreen(Kind: TScreenKind);
    procedure ClearItems;
    procedure AddItem(const ItemCaption: string);
    procedure LayoutItems;
    procedure SetFocusIndex(NewIndex: Integer);
    procedure MoveFocus(DX, DY: Integer);
    procedure ActivateFocused;
    procedure GoBack;
    procedure GoHome;
    procedure HandleEvent(const EventName: string);
    procedure Log(const Text: string);
    procedure WMGestureEvent(var Msg: TMessage); message WM_GESTURE_EVENT;
    procedure WMEngineExited(var Msg: TMessage); message WM_ENGINE_EXITED;
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

const
  FOCUS_COLOR = clYellow;         // 포커스 항목 배경 - 전맹 외 저시력 사용자 대비
  NORMAL_COLOR = clBtnFace;
  DOC_NAMES: array[0..3] of string =
    ('주민등록등본', '가족관계증명서', '건축물대장', '토지(임야)대장');

{ ───── 폼 수명 ───── }

procedure TForm1.FormCreate(Sender: TObject);
begin
  Caption := '제스처 키오스크 데모 - 파이프 수신 (gesture_kiosk)';
  FItems := TList.Create;
  FEngine := TEngineProcess.Create;

  FTitleLabel := TLabel.Create(Self);
  FTitleLabel.Parent := Self;
  FTitleLabel.AutoSize := False;
  FTitleLabel.SetBounds(0, 24, ClientWidth, 40);
  FTitleLabel.Alignment := taCenter;
  FTitleLabel.Font.Size := 18;
  FTitleLabel.Font.Style := [fsBold];

  BuildScreen(skMenu);
  Log('[안내] 키보드로도 시험 가능 - 방향키=포커스, Enter=ok, Backspace=back, Home=home');
  StartEngine;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  RestartTimer.Enabled := False;
  FEngine.Stop;                    // UI 종료 = 엔진도 종료 (가이드 §4 주의)
  FEngine.Free;
  FItems.Free;
end;

{ ───── 엔진 실행·재기동 ───── }

procedure TForm1.StartEngine;
var
  WorkDir: string;
begin
  WorkDir := ExpandFileName(ExtractFilePath(Application.ExeName) + ENGINE_WORKDIR);
  FEngineAlive := FEngine.Start(ENGINE_COMMAND, WorkDir, Handle);
  if FEngineAlive then
  begin
    StatusBar.SimpleText := ' 엔진 실행 중 - ' + ENGINE_COMMAND + '  (작업 폴더: ' + WorkDir + ')';
    Log('[엔진] 실행: ' + ENGINE_COMMAND);
  end
  else
  begin
    StatusBar.SimpleText := ' 엔진 실행 실패 - MainForm.pas의 ENGINE_COMMAND·ENGINE_WORKDIR 확인';
    Log('[엔진] 실행 실패 - 경로 설정 확인 (3초 후 재시도)');
  end;
end;

procedure TForm1.RestartTimerTimer(Sender: TObject);
begin
  if not FEngineAlive then
  begin
    FEngine.Stop;
    StartEngine;                   // 엔진 사망·실행 실패 - 주기 재기동
  end;
end;

procedure TForm1.WMEngineExited(var Msg: TMessage);
begin
  FEngineAlive := False;
  StatusBar.SimpleText := ' 엔진 종료됨 - 3초 안에 자동 재실행';
  Log('[엔진] 종료 감지(EOF) - 자동 재실행 대기');
end;

{ ───── 이벤트 수신·분기 ───── }

procedure TForm1.WMGestureEvent(var Msg: TMessage);
var
  Raw: PChar;
  Line, Rest, EventName: string;
  Sep: Integer;
begin
  Raw := PChar(Msg.LParam);
  Line := StrPas(Raw);
  StrDispose(Raw);                 // EngineProcess가 StrNew로 넘긴 소유권 회수

  Log('[수신] ' + Line);
  Rest := Line;
  Sep := Pos('|', Rest);
  if (Sep = 0) or (Copy(Rest, 1, Sep - 1) <> 'GESTURE') then Exit;
  Delete(Rest, 1, Sep);
  Sep := Pos('|', Rest);
  if Sep = 0 then Exit;
  EventName := Copy(Rest, 1, Sep - 1);
  HandleEvent(EventName);
end;

procedure TForm1.HandleEvent(const EventName: string);
begin
  // 이벤트 7종 (연동가이드 §3) - 사용자 기준 방향 그대로 포커스를 옮긴다
  if EventName = 'left' then MoveFocus(-1, 0)
  else if EventName = 'right' then MoveFocus(1, 0)
  else if EventName = 'top' then MoveFocus(0, -1)
  else if EventName = 'bottom' then MoveFocus(0, 1)
  else if EventName = 'ok' then ActivateFocused
  else if EventName = 'back' then GoBack
  else if EventName = 'home' then GoHome
  else Log('[무시] 알 수 없는 이벤트: ' + EventName);
end;

procedure TForm1.FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
begin
  // 키보드 폴백 - 엔진 없이 포커스·화면 로직만 시험할 때
  case Key of
    VK_LEFT:   HandleEvent('left');
    VK_RIGHT:  HandleEvent('right');
    VK_UP:     HandleEvent('top');
    VK_DOWN:   HandleEvent('bottom');
    VK_RETURN: HandleEvent('ok');
    VK_BACK:   HandleEvent('back');
    VK_HOME:   HandleEvent('home');
  end;
end;

{ ───── 화면 구성 ───── }

procedure TForm1.BuildScreen(Kind: TScreenKind);
begin
  FScreenKind := Kind;
  ClearItems;
  case Kind of
    skMenu:
      begin
        FTitleLabel.Caption := '발급할 증명서를 선택하세요';
        FCols := 2;                              // 2x2 격자 - 상하좌우 이동 시연
        AddItem(DOC_NAMES[0]);
        AddItem(DOC_NAMES[1]);
        AddItem(DOC_NAMES[2]);
        AddItem(DOC_NAMES[3]);
      end;
    skConfirm:
      begin
        FTitleLabel.Caption := FSelectedDoc + ' - 발급할까요?';
        FCols := 2;                              // 1행 2열
        AddItem('발급하기');
        AddItem('취소');
      end;
    skDone:
      begin
        FTitleLabel.Caption := FSelectedDoc + ' 발급이 완료되었습니다';
        FCols := 1;
        AddItem('처음으로');
      end;
  end;
  LayoutItems;
  SetFocusIndex(0);
end;

procedure TForm1.ClearItems;
var
  I: Integer;
begin
  for I := 0 to FItems.Count - 1 do
    TPanel(FItems[I]).Free;
  FItems.Clear;
end;

procedure TForm1.AddItem(const ItemCaption: string);
var
  Panel: TPanel;
begin
  Panel := TPanel.Create(Self);
  Panel.Parent := Self;
  Panel.Caption := ItemCaption;
  Panel.Font.Size := 14;
  Panel.BevelOuter := bvRaised;
  Panel.Color := NORMAL_COLOR;
  FItems.Add(Panel);
end;

procedure TForm1.LayoutItems;
const
  ITEM_W = 280; ITEM_H = 110; GAP = 28; TOP_BASE = 110;
var
  I, Rows, Col, Row, LeftBase: Integer;
begin
  Rows := (FItems.Count + FCols - 1) div FCols;
  LeftBase := (ClientWidth - FCols * ITEM_W - (FCols - 1) * GAP) div 2;
  for I := 0 to FItems.Count - 1 do
  begin
    Col := I mod FCols;
    Row := I div FCols;
    TPanel(FItems[I]).SetBounds(LeftBase + Col * (ITEM_W + GAP),
                                TOP_BASE + Row * (ITEM_H + GAP), ITEM_W, ITEM_H);
  end;
  if Rows = 0 then Exit;   // 방어 - 빈 화면은 없다
end;

{ ───── 포커스 이동 (회사 UI에 들어갈 핵심 로직) ───── }

procedure TForm1.SetFocusIndex(NewIndex: Integer);
var
  I: Integer;
  Panel: TPanel;
begin
  FFocusIndex := NewIndex;
  for I := 0 to FItems.Count - 1 do
  begin
    Panel := TPanel(FItems[I]);
    if I = FFocusIndex then
    begin
      Panel.Color := FOCUS_COLOR;
      Panel.Font.Style := [fsBold];
    end
    else
    begin
      Panel.Color := NORMAL_COLOR;
      Panel.Font.Style := [];
    end;
  end;
  // 실제 화면낭독 키오스크라면 여기서 포커스 항목을 TTS로 낭독한다 (UI 담당 - №7)
  Log('[포커스] ' + TPanel(FItems[FFocusIndex]).Caption);
end;

procedure TForm1.MoveFocus(DX, DY: Integer);
var
  Col, Row, Rows, NewIndex: Integer;
begin
  Col := FFocusIndex mod FCols;
  Row := FFocusIndex div FCols;
  Rows := (FItems.Count + FCols - 1) div FCols;
  Col := Col + DX;
  Row := Row + DY;
  // 격자 밖은 이동하지 않는다(랩 없음) - 스크린리더식 탐색은 예측 가능성이 우선
  if (Col < 0) or (Col >= FCols) or (Row < 0) or (Row >= Rows) then
  begin
    Log('[포커스] 끝 - 이동 없음');
    Exit;
  end;
  NewIndex := Row * FCols + Col;
  if NewIndex >= FItems.Count then
  begin
    Log('[포커스] 빈 칸 - 이동 없음');
    Exit;
  end;
  SetFocusIndex(NewIndex);
end;

procedure TForm1.ActivateFocused;
var
  ItemCaption: string;
begin
  ItemCaption := TPanel(FItems[FFocusIndex]).Caption;
  Log('[실행] ' + ItemCaption);
  case FScreenKind of
    skMenu:
      begin
        FSelectedDoc := ItemCaption;
        BuildScreen(skConfirm);
      end;
    skConfirm:
      if ItemCaption = '발급하기' then
        BuildScreen(skDone)
      else
        BuildScreen(skMenu);
    skDone:
      BuildScreen(skMenu);
  end;
end;

procedure TForm1.GoBack;
begin
  case FScreenKind of
    skMenu: Log('[무시] 이미 처음 화면 - back 없음');   // 화면 이탈 사고 방지
    skConfirm: BuildScreen(skMenu);
    skDone: BuildScreen(skMenu);
  end;
end;

procedure TForm1.GoHome;
begin
  if FScreenKind = skMenu then
    Log('[무시] 이미 처음 화면 - home 없음')
  else
    BuildScreen(skMenu);
end;

{ ───── 로그 ───── }

procedure TForm1.Log(const Text: string);
begin
  LogMemo.Lines.Add(FormatDateTime('hh:nn:ss', Now) + '  ' + Text);
  // 데모 편의 - 로그가 너무 길어지면 앞부분을 버린다
  while LogMemo.Lines.Count > 300 do
    LogMemo.Lines.Delete(0);
end;

end.
