
param (
    [switch] $benchmark = $false
)

Push-Location $PSScriptRoot

$test_files = (
    "archiviz/archiviz.blend",
    "barbershop/barbershop_interior_cpu.blend",
    "blender_splash_fishy_cat/fishy_cat.blend",
    "classroom/classroom.blend",
    "forest/forest.blend",
    "mr_elephant/mr_elephant.blend",
    "pabellon_barcelona/pavillon_barcelone_v1_2.blend",
    "pokedstudio/splash-pokedstudio.blend",
    "race_spaceship/race_spaceship.blend",
    "stylized_levi/stylized_levi.blend",
    "temple/temple.blend",
    "the_junk_shop/the_junk_shop.blend",
    "tree_creature/tree_creature.blend",
    "wanderer/wanderer.blend",
    "wasp_bot/wasp_bot.blend"
)

$args = @()

if ($benchmark) {
    $args += "-benchmark"
}
$env:blender_args = $args

Measure-Command -Expression {
    foreach ($file in $test_files) {
        blender -b $file -P test_datasmith_export.py -- $env:blender_args
    }
}

# best time: 9:05, intel i7 4790k


Pop-Location
