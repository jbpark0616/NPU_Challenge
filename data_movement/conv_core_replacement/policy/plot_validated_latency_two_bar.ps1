param(
    [string]$OutputSvg = "$PSScriptRoot/figures/validated_remapping_latency_two_bar.svg"
)

$ErrorActionPreference = "Stop"
$culture = [System.Globalization.CultureInfo]::InvariantCulture

# Distinct, successfully completed stock and remapped full-model captures.
# Metric: sum of each device operation's TRISC critical span at 650 MHz.
$stockMs = 2.056535
$remappedMs = 2.049374
$deltaPercent = 100.0 * ($remappedMs - $stockMs) / $stockMs
$deltaUs = 1000.0 * ($remappedMs - $stockMs)

function F([double]$value, [string]$format) {
    return $value.ToString($format, $culture)
}

$chartTop = 105
$chartBottom = 555
$chartHeight = $chartBottom - $chartTop
$axisMax = 2.2
$stockHeight = $chartHeight * $stockMs / $axisMax
$remappedHeight = $chartHeight * $remappedMs / $axisMax
$stockY = $chartBottom - $stockHeight
$remappedY = $chartBottom - $remappedHeight

$svg = @"
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
<style>
  text { font-family: Arial, Helvetica, sans-serif; fill: #17212B; }
  .tick { font-size: 18px; fill: #65717C; }
  .axis-label { font-size: 22px; font-weight: 700; }
  .bar-label { font-size: 24px; font-weight: 700; }
  .value { font-size: 30px; font-weight: 700; }
  .delta { font-size: 26px; font-weight: 700; fill: #1477A8; }
  .delta-sub { font-size: 18px; fill: #56636E; }
</style>

<line x1="190" y1="$chartBottom" x2="1050" y2="$chartBottom" stroke="#17212B" stroke-width="2"/>
<line x1="190" y1="$chartTop" x2="190" y2="$chartBottom" stroke="#17212B" stroke-width="2"/>

<line x1="190" y1="$chartBottom" x2="1050" y2="$chartBottom" stroke="#D6DCE1" stroke-width="1"/>
<line x1="190" y1="$($chartBottom - $chartHeight * 0.25)" x2="1050" y2="$($chartBottom - $chartHeight * 0.25)" stroke="#D6DCE1" stroke-width="1"/>
<line x1="190" y1="$($chartBottom - $chartHeight * 0.50)" x2="1050" y2="$($chartBottom - $chartHeight * 0.50)" stroke="#D6DCE1" stroke-width="1"/>
<line x1="190" y1="$($chartBottom - $chartHeight * 0.75)" x2="1050" y2="$($chartBottom - $chartHeight * 0.75)" stroke="#D6DCE1" stroke-width="1"/>
<line x1="190" y1="$chartTop" x2="1050" y2="$chartTop" stroke="#D6DCE1" stroke-width="1"/>

<text x="170" y="$($chartBottom + 6)" class="tick" text-anchor="end">0</text>
<text x="170" y="$($chartBottom - $chartHeight * 0.25 + 6)" class="tick" text-anchor="end">0.55</text>
<text x="170" y="$($chartBottom - $chartHeight * 0.50 + 6)" class="tick" text-anchor="end">1.10</text>
<text x="170" y="$($chartBottom - $chartHeight * 0.75 + 6)" class="tick" text-anchor="end">1.65</text>
<text x="170" y="$($chartTop + 6)" class="tick" text-anchor="end">2.20</text>
<text x="58" y="330" class="axis-label" text-anchor="middle" transform="rotate(-90 58 330)">Aggregate device latency (ms)</text>

<rect x="350" y="$(F $stockY '0.0')" width="230" height="$(F $stockHeight '0.0')" rx="7" fill="#D55E00"/>
<rect x="695" y="$(F $remappedY '0.0')" width="230" height="$(F $remappedHeight '0.0')" rx="7" fill="#1477A8"/>
<text x="465" y="$(F ($stockY - 18) '0.0')" class="value" text-anchor="middle">$(F $stockMs '0.000') ms</text>
<text x="810" y="$(F ($remappedY - 18) '0.0')" class="value" text-anchor="middle">$(F $remappedMs '0.000') ms</text>
<text x="465" y="605" class="bar-label" text-anchor="middle">Stock</text>
<text x="810" y="605" class="bar-label" text-anchor="middle">Validated remapping</text>

<line x1="465" y1="58" x2="810" y2="58" stroke="#1477A8" stroke-width="2"/>
<line x1="465" y1="58" x2="465" y2="78" stroke="#1477A8" stroke-width="2"/>
<line x1="810" y1="58" x2="810" y2="78" stroke="#1477A8" stroke-width="2"/>
<text x="637.5" y="43" class="delta" text-anchor="middle">Δ $(F $deltaPercent '+0.00;-0.00;0.00')%</text>
<text x="637.5" y="82" class="delta-sub" text-anchor="middle">$(F $deltaUs '+0.0;-0.0;0.0') µs · no measurable overhead</text>
</svg>
"@

$outputDirectory = Split-Path -Parent $OutputSvg
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
[System.IO.File]::WriteAllText($OutputSvg, $svg, [System.Text.UTF8Encoding]::new($false))
Write-Host "Stock: $(F $stockMs '0.000000') ms"
Write-Host "Remapped: $(F $remappedMs '0.000000') ms"
Write-Host "Delta: $(F $deltaPercent '+0.000;-0.000;0.000')%"
Write-Host "Wrote $OutputSvg"
