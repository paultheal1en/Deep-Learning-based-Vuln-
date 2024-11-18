static bool ranges_can_merge(Range *range1, Range *range2)

{

    return !(range1->end < range2->begin || range2->end < range1->begin);

}
