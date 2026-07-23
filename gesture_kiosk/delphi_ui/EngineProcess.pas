unit EngineProcess;

{ 제스처 엔진 실행 + 파이프(stdio) 수신 (2026-07-23 - 파이프 연동 확정).

  구조: CreatePipe(익명 파이프) -> CreateProcess(엔진 stdout을 파이프에 연결)
  -> 수신 스레드가 ReadFile로 읽어 CRLF 단위로 잘라 메인 폼에 PostMessage.
  UI는 절대 이 유닛(수신 스레드)에서 만지지 않는다 - 메시지로만 전달.

  호환(2026-07-23 2차): 델파이7과 최신 델파이(Community Edition 포함) 양쪽에서
  컴파일되도록 파이프 바이트 처리를 전부 AnsiChar/AnsiString으로 명시했다 -
  델파이 2009+는 Char/string이 2바이트(유니코드)라, 그대로 두면 ReadFile
  바이트 수와 문자 수가 어긋난다. 이벤트 줄은 ASCII라 Ansi로 충분하다.
  문자열 전달도 StrNew/StrDispose(버전별 소속 유닛이 다름) 대신
  GetMem/FreeMem으로 통일 - 수신 측은 FreeMem으로 해제한다.

  규칙:
  - 부모 쪽 파이프 "쓰기 끝"은 CreateProcess 직후 닫는다 - 안 닫으면 엔진이
    죽어도 EOF가 오지 않아 스레드가 영원히 기다린다.
  - 엔진 stderr(로그)는 NUL로 버린다 - 로그는 엔진의 logs/ 파일에 남는다.
    stdout은 이벤트 전용 채널(GESTURE| 줄)이다. }

interface

uses
  Windows, Messages, Classes, SysUtils;

const
  WM_GESTURE_EVENT = WM_USER + 101;   // LParam = PAnsiChar 한 줄 (수신 측이 FreeMem)
  WM_ENGINE_EXITED = WM_USER + 102;   // 엔진 종료(EOF) 알림 - 재기동 판단용

type
  TReadThread = class(TThread)
  private
    FPipe: THandle;
    FNotify: HWND;
  protected
    procedure Execute; override;
  public
    constructor Create(APipe: THandle; ANotify: HWND);
  end;

  TEngineProcess = class
  private
    FProcessInfo: TProcessInformation;
    FReadPipe: THandle;
    FRunning: Boolean;
  public
    function Start(const CommandLine, WorkDir: string; Notify: HWND): Boolean;
    procedure Stop;
    property IsRunning: Boolean read FRunning;
  end;

implementation

function NewAnsiLine(const S: AnsiString): PAnsiChar;
{ 수신 줄 1개를 힙에 복사 - PostMessage로 스레드 경계를 넘기기 위해.
  수신 측(메인 폼)이 FreeMem으로 해제한다 (소유권 이전). }
begin
  GetMem(Result, Length(S) + 1);
  if Length(S) > 0 then
    Move(S[1], Result^, Length(S));
  Result[Length(S)] := #0;
end;

{ TReadThread }

constructor TReadThread.Create(APipe: THandle; ANotify: HWND);
begin
  FPipe := APipe;
  FNotify := ANotify;
  FreeOnTerminate := True;
  inherited Create(False);
end;

procedure TReadThread.Execute;
var
  Buf: array[0..1023] of AnsiChar;   // 바이트 버퍼 - 신형 델파이에서도 1바이트 유지
  Got: Cardinal;
  Acc, Line: AnsiString;
  P: Integer;
begin
  Acc := '';
  while ReadFile(FPipe, Buf, SizeOf(Buf), Got, nil) and (Got > 0) do
  begin
    SetString(Line, PAnsiChar(@Buf[0]), Integer(Got));
    Acc := Acc + Line;
    P := Pos(AnsiString(#13#10), Acc);
    while P > 0 do
    begin
      Line := Copy(Acc, 1, P - 1);
      Delete(Acc, 1, P + 1);
      if Line <> '' then
        PostMessage(FNotify, WM_GESTURE_EVENT, 0, LPARAM(NewAnsiLine(Line)));
      P := Pos(AnsiString(#13#10), Acc);
    end;
  end;
  PostMessage(FNotify, WM_ENGINE_EXITED, 0, 0);
end;

{ TEngineProcess }

function TEngineProcess.Start(const CommandLine, WorkDir: string; Notify: HWND): Boolean;
var
  SA: TSecurityAttributes;
  SI: TStartupInfo;
  WritePipe, NulHandle: THandle;
  Cmd: string;
begin
  Result := False;
  SA.nLength := SizeOf(SA);
  SA.bInheritHandle := True;
  SA.lpSecurityDescriptor := nil;
  if not CreatePipe(FReadPipe, WritePipe, @SA, 0) then Exit;
  // 읽기 끝은 자식에게 상속 금지 - 자식이 물고 있으면 종료 후에도 EOF가 안 온다
  SetHandleInformation(FReadPipe, HANDLE_FLAG_INHERIT, 0);

  NulHandle := CreateFile('NUL', GENERIC_WRITE, FILE_SHARE_WRITE, @SA,
                          OPEN_EXISTING, 0, 0);

  FillChar(SI, SizeOf(SI), 0);
  SI.cb := SizeOf(SI);
  SI.dwFlags := STARTF_USESTDHANDLES or STARTF_USESHOWWINDOW;
  SI.hStdInput := 0;
  SI.hStdOutput := WritePipe;      // 엔진 stdout -> 파이프 (이벤트 전용)
  SI.hStdError := NulHandle;       // 로그(stderr)는 버림 - 파일 로그가 따로 있다
  SI.wShowWindow := SW_HIDE;

  Cmd := CommandLine;              // CreateProcess는 쓰기 가능한 버퍼를 요구한다
  UniqueString(Cmd);
  Result := CreateProcess(nil, PChar(Cmd), nil, nil, True, CREATE_NO_WINDOW,
                          nil, PChar(WorkDir), SI, FProcessInfo);

  CloseHandle(WritePipe);          // 부모 쪽 쓰기 끝 닫기 (유닛 머리 주석 참고)
  if NulHandle <> INVALID_HANDLE_VALUE then CloseHandle(NulHandle);

  if not Result then
  begin
    CloseHandle(FReadPipe);
    Exit;
  end;
  FRunning := True;
  TReadThread.Create(FReadPipe, Notify);   // FreeOnTerminate - 스스로 정리
end;

procedure TEngineProcess.Stop;
begin
  if not FRunning then Exit;
  FRunning := False;
  TerminateProcess(FProcessInfo.hProcess, 0);
  CloseHandle(FProcessInfo.hProcess);
  CloseHandle(FProcessInfo.hThread);
  CloseHandle(FReadPipe);          // ReadFile 해제 -> 수신 스레드 종료
end;

end.
