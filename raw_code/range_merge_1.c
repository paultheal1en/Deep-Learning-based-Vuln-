static void range_merge(Range *range1, Range *range2)

{

    if (range1->end < range2->end) {

        range1->end = range2->end;

    }

    if (range1->begin > range2->begin) {

        range1->begin = range2->begin;

    }

}
