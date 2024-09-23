Sub Validate()

    Dim lastRow As Long
    Dim i As Long
    Dim errorCount As Long
    Dim inputError As Boolean
    Dim combinationError As Boolean
    Dim totalError As Long
    Dim ws As Worksheet
    Dim valid As Boolean
    
    Set ws = ThisWorkbook.Sheets("Sheet1")
    
    lastRow = ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
    totalError = 0
    errorCount = 0
    valid = True
    
    For i = 2 To lastRow
        
        inputError = False
        combinationError = False
        
        ' Check for invalid combinations in column A (based on the rules in the instructions)
        Select Case ws.Cells(i, 1).Value
            Case "Orifice"
                If ws.Cells(i + 1, 1).Value = "Orifice" Or ws.Cells(i + 1, 1).Value = "90DegBend" Then
                    combinationError = True
                End If
            Case "90degbend"
                If ws.Cells(i + 1, 1).Value = "Orifice" Or ws.Cells(i + 1, 1).Value = "90DegBend" Then
                    combinationError = True
                End If
            Case "chambervol"
                If ws.Cells(i + 1, 1).Value = "chambervol" Then
                    combinationError = True
                End If
        End Select
        
        If combinationError Then
            ws.Cells(i, 1).Interior.Color = vbRed
            errorCount = errorCount + 1
        Else
            ws.Cells(i, 1).Interior.Color = vbWhite
        End If
        
        ' Check for missing input values
        If ws.Cells(i, 1).Value <> "90DegBend" Then
            If ws.Cells(i, 3).Value = "" Or ws.Cells(i, 5).Value = "" Then
                inputError = True
            End If
        Else
            If ws.Cells(i, 3).Value = "" Then
                inputError = True
            End If
        End If
        
        If inputError Then
            ws.Cells(i, 4).Interior.Color = vbRed
            totalError = totalError + 1
        Else
            ws.Cells(i, 4).Interior.Color = vbWhite
        End If
    
    Next i
    
    ' Display total errors in G column
    ws.Cells(2, 7).Value = totalError
    
    ' If total errors exist, highlight the Validate button area
    If totalError > 0 Or errorCount > 0 Then
        MsgBox "Errors detected! Please correct the input and try again.", vbExclamation
        ws.Range("G1:H1").Interior.Color = vbRed ' Optional: Change color of the Validate button area
    Else
        MsgBox "All inputs are valid. You can proceed!", vbInformation
        ws.Range("G1:H1").Interior.Color = vbGreen ' Optional: Success color
    End If
    
End Sub

