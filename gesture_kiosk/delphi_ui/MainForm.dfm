object Form1: TForm1
  Left = 200
  Top = 120
  Width = 820
  Height = 660
  Caption = 'Gesture Kiosk Demo'
  Color = clBtnFace
  Font.Charset = HANGEUL_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Gulim'
  Font.Style = []
  KeyPreview = True
  OldCreateOrder = False
  Position = poScreenCenter
  OnClose = FormClose
  OnCreate = FormCreate
  OnKeyDown = FormKeyDown
  PixelsPerInch = 96
  TextHeight = 13
  object LogMemo: TMemo
    Left = 0
    Top = 464
    Width = 812
    Height = 150
    Align = alBottom
    Color = clWhite
    Font.Charset = HANGEUL_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'Gulim'
    Font.Style = []
    ParentFont = False
    ReadOnly = True
    ScrollBars = ssVertical
    TabOrder = 0
  end
  object StatusBar: TStatusBar
    Left = 0
    Top = 614
    Width = 812
    Height = 19
    Panels = <>
    SimplePanel = True
  end
  object RestartTimer: TTimer
    Interval = 3000
    OnTimer = RestartTimerTimer
    Left = 16
    Top = 16
  end
end
