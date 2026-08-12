"""Test-set concept names and coarse category labels for similarity matrices."""

from __future__ import annotations

from typing import Dict, List, Tuple

CATEGORY_ORDER = ['animal', 'food', 'vehicle', 'tool', 'others']

_ANIMAL = {
    'antelope', 'beaver', 'bug', 'cat', 'caterpillar', 'cheetah', 'cobra', 'crab', 'crow',
    'dalmatian', 'dragonfly', 'eagle', 'eel', 'elephant', 'flamingo', 'goose', 'gopher',
    'gorilla', 'grasshopper', 'hummingbird', 'lamb', 'lightning_bug', 'manatee', 'mosquito',
    'mussel', 'ostrich', 'panther', 'pheasant', 'piglet', 'possum', 'pug', 'rhinoceros',
    'rooster', 'seagull', 'tick', 'turkey',
}
_FOOD = {
    'banana', 'basil', 'batter', 'birthday_cake', 'bok_choy', 'bread', 'bun', 'calamari',
    'cashew', 'cheese', 'coconut', 'coffee_bean', 'cookie', 'cordon_bleu', 'creme_brulee',
    'crepe', 'croissant', 'crumb', 'cupcake', 'dessert', 'egg', 'espresso', 'fruit', 'garlic',
    'hamburger', 'jelly_bean', 'lettuce', 'meatloaf', 'okra', 'omelet', 'onion', 'orange',
    'pear', 'pepper1', 'pie', 'popcorn', 'popsicle', 'pretzel', 'radish', 'raspberry',
    'sausage', 'scallion', 'scallop', 'seaweed', 'strawberry', 'tomato_sauce', 'walnut',
    'wheat', 'wine', 'marijuana', 'seed',
}
_VEHICLE = {
    'aircraft_carrier', 'bike', 'boat', 'buggy', 'cart', 'cruise_ship', 'ferry', 'golf_cart',
    'gondola', 'jeep', 'minivan', 'sailboat', 'scooter', 'skateboard', 'sled', 'station_wagon',
    'submarine', 'unicycle', 'wheelchair',
}
_TOOL = {
    'backscratcher', 'baseball_bat', 'baton4', 'blowtorch', 'bottle_opener', 'brace', 'chain',
    'chime', 'chopsticks', 'cleat', 'cleaver', 'dagger', 'fork', 'hammer', 'handbrake', 'kettle',
    'ladle', 'metal_detector', 'pickax', 'pocketknife', 'sandpaper', 'slingshot', 'spatula',
    'spoon', 'tongs', 'tool', 'vise', 'wok', 'stethoscope',
}


def concept_from_relpath(relpath: str) -> str:
    """e.g. test_images/00005_banana/banana_09s.jpg -> banana"""
    folder = relpath.split('/')[1]
    return folder.split('_', 1)[1]


def category_for_concept(concept: str) -> str:
    if concept in _ANIMAL:
        return 'animal'
    if concept in _FOOD:
        return 'food'
    if concept in _VEHICLE:
        return 'vehicle'
    if concept in _TOOL:
        return 'tool'
    return 'others'


def sort_indices_by_category(concepts: List[str]) -> Tuple[List[int], List[str], Dict[str, int]]:
    """Return row/col order, per-index category labels, and block sizes."""
    categories = [category_for_concept(c) for c in concepts]
    order = sorted(
        range(len(concepts)),
        key=lambda i: (CATEGORY_ORDER.index(categories[i]), i),
    )
    sorted_cats = [categories[i] for i in order]
    block_sizes: Dict[str, int] = {k: 0 for k in CATEGORY_ORDER}
    for c in sorted_cats:
        block_sizes[c] += 1
    return order, sorted_cats, block_sizes
