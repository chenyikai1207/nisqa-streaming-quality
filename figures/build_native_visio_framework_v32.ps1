param([string]$SourceJson,[string]$OutputFolder)
$ErrorActionPreference='Stop'
$spec=Get-Content -LiteralPath $SourceJson -Raw -Encoding UTF8|ConvertFrom-Json
$app=New-Object -ComObject Visio.Application
$app.Visible=$false
$app.AlertResponse=7
$doc=$null
function Add-TextShape($page,[double]$left,[double]$top,[double]$width,[double]$height,[string]$text,[double]$size,[bool]$center,[string]$name) {
  $canvasH=[double]$spec.canvas_mm[1]
  $s=$page.DrawRectangle($left/25.4,($canvasH-$top-$height)/25.4,($left+$width)/25.4,($canvasH-$top)/25.4)
  $s.NameU=$name
  $s.Text=$text
  $s.CellsU('LinePattern').FormulaU='0'
  $s.CellsU('FillPattern').FormulaU='0'
  $s.CellsU('Char.Font').FormulaU='FONT("华文中宋")'
  $s.CellsU('Char.Size').FormulaU=([string]$size)+' pt'
  $s.CellsU('Char.Color').FormulaU='RGB(0,0,0)'
  $s.CellsU('Para.HorzAlign').FormulaU=$(if($center){'1'}else{'0'})
  $s.CellsU('VerticalAlign').FormulaU='1'
  foreach($cell in @('LeftMargin','RightMargin','TopMargin','BottomMargin')) {$s.CellsU($cell).FormulaU='0 mm'}
  return $s
}
try {
  $doc=$app.Documents.Add('')
  $page=$doc.Pages.Item(1)
  $page.Name='模型框架'
  $page.PageSheet.CellsU('PageWidth').FormulaU=([string]$spec.canvas_mm[0])+' mm'
  $page.PageSheet.CellsU('PageHeight').FormulaU=([string]$spec.canvas_mm[1])+' mm'
  $labels=@()
  foreach($n in $spec.nodes) {
    $left=[double]$n.x_mm;$top=[double]$n.y_mm;$width=[double]$n.width_mm;$height=[double]$n.height_mm;$canvasH=[double]$spec.canvas_mm[1]
    $box=$page.DrawRectangle($left/25.4,($canvasH-$top-$height)/25.4,($left+$width)/25.4,($canvasH-$top)/25.4)
    $box.NameU='box_'+$n.id
    $box.CellsU('FillPattern').FormulaU='0';$box.CellsU('LineColor').FormulaU='RGB(0,0,0)';$box.CellsU('LineWeight').FormulaU='0.75 pt'
    $labels+=Add-TextShape $page ($left+1) ($top+1) ($width-2) 7 ([string]$n.title) 9 $true ('title_'+$n.id)
    $labels+=Add-TextShape $page ($left+1) ($top+8) ($width-2) ($height-9) ([string]$(if($n.id -eq 'front_state'){'滤波与窗口'+[char]10+'重采样历史'}else{$n.body})) 8.3 $true ('body_'+$n.id)
  }
  $edgeIndex=0
  foreach($e in $spec.edges) {
    $edgeIndex++
    $points=$(if($e.points_mm){$e.points_mm}else{@($e.start_mm,$e.end_mm)})
    for($k=0;$k -lt ($points.Count-1);$k++) {
      $a=$points[$k];$b=$points[$k+1]
      $line=$page.DrawLine([double]$a[0]/25.4,([double]$spec.canvas_mm[1]-[double]$a[1])/25.4,[double]$b[0]/25.4,([double]$spec.canvas_mm[1]-[double]$b[1])/25.4)
      $line.NameU='edge_'+$edgeIndex+'_'+$k
      $line.CellsU('LineColor').FormulaU='RGB(0,0,0)';$line.CellsU('LineWeight').FormulaU='0.75 pt'
      $line.CellsU('LinePattern').FormulaU=$(if($e.dashed){'2'}else{'1'})
      if($k -eq ($points.Count-2)){$line.CellsU('EndArrow').FormulaU='4';$line.CellsU('EndArrowSize').FormulaU='1'}
    }
  }
  $freeIndex=0
  foreach($n in $spec.free_text) {
    $freeIndex++
    $labels+=Add-TextShape $page ([double]$n.x_mm) ([double]$n.y_mm-4) ([double]$spec.canvas_mm[0]-[double]$n.x_mm-2) 8 ([string]$n.text) 8 $false ('note_'+$freeIndex)
  }
  $stem=Join-Path $OutputFolder 'fig1_framework_native_zh_v32'
  $doc.SaveAs($stem+'.vsdx')
  $page.Export($stem+'.svg')
  $page.Export($stem+'.emf')
  $doc.ExportAsFixedFormat(1,$stem+'.pdf',1,0)
  $nativeTexts=@($labels|ForEach-Object {$_.Text})
  $receipt=[ordered]@{time=(Get-Date).ToString('o');status='ACTUAL_NATIVE_VISIO_FRAMEWORK_CREATED';vsdx=$stem+'.vsdx';svg=$stem+'.svg';emf=$stem+'.emf';pdf=$stem+'.pdf';source_json=$SourceJson;native_text_shapes=$labels.Count;native_shape_count=$page.Shapes.Count;labels=$nativeTexts;native_visio=$app.Version;raw4_before_normalization_to_output_head=$spec.raw4_scientific_connection;new_scientific_calculations=0}
  [System.IO.File]::WriteAllText((Join-Path $OutputFolder 'NATIVE_VISIO_FIG1_v32_RECEIPT.json'),($receipt|ConvertTo-Json -Depth 8),[System.Text.UTF8Encoding]::new($false))
  $doc.Close()
  $receipt|ConvertTo-Json -Depth 5
} finally {if($doc){try {$doc.Close()} catch{}};try {$app.Quit()}catch{};[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)}