object Form1: TForm1
  Left = 410
  Top = 98
  Width = 820
  Height = 660
  Caption = 'Gesture Kiosk Demo'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Tahoma'
  Font.Style = []
  KeyPreview = True
  OldCreateOrder = False
  Position = poScreenCenter
  OnClose = FormClose
  OnCreate = FormCreate
  OnKeyDown = FormKeyDown
  PixelsPerInch = 96
  TextHeight = 14
  object LogMemo: TMemo
    Left = 0
    Top = 441
    Width = 804
    Height = 161
    Align = alBottom
    Color = clWhite
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -12
    Font.Name = 'Tahoma'
    Font.Style = []
    ParentFont = False
    ReadOnly = True
    ScrollBars = ssVertical
    TabOrder = 0
  end
  object StatusBar: TStatusBar
    Left = 0
    Top = 602
    Width = 804
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
