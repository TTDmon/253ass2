#!/bin/bash

# dir
mkdir -p output/generated_batch

# chords
progressions=(
    "C:maj7 A:min7 D:min7 G7"
    "F:maj7 G:7 E:min7 A:7"
    "D:min7 G:7 C:maj7 A:min7"
    "E:min7 A:7 D:min7 G:7"
    "C:7 F:7 G:7 C:7"
    "A:min7 D:min7 G:maj7 C:maj7"
    "G:maj D:maj A:maj E:maj"
    "F:maj7 E:min7 D:min7 C:maj7"
    "B:min7 E:7 A:maj7 F#:min7"
    "D:maj B:min G:maj A:maj"
)

for i in {0..99}
do
    # randomly select a chord progression
    idx=$((RANDOM % ${#progressions[@]}))
    chords="${progressions[$idx]}"

    echo "🎵 Generating MIDI $i with chords: $chords"

    python generate_v2.py \
        --chords $chords \
        --output output/generated_batch/generated_$i.mid \
done
