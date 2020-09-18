[CmdletBinding(SupportsShouldProcess=$True)]
param(
)

if($PSCmdlet.ShouldProcess('dotnet paket', 'Checking and downloading dependencies')) {
    dotnet tool restore
    dotnet paket restore
}
