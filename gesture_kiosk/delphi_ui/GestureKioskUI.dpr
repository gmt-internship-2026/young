program GestureKioskUI;

{ 제스처 키오스크 데모 UI - 파이프(stdio) 수신 + 포커스 이동 (2026-07-23).
  빌드·실행 방법: 같은 폴더의 빌드_실행_안내.md 참고. }

uses
  Forms,
  EngineProcess in 'EngineProcess.pas',
  MainForm in 'MainForm.pas' {Form1};

{$R *.res}

begin
  Application.Initialize;
  Application.Title := 'Gesture Kiosk Demo';
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
