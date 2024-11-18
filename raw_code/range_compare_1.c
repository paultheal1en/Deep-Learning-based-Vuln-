static gint range_compare(gconstpointer a, gconstpointer b)

{

    Range *ra = (Range *)a, *rb = (Range *)b;

    if (ra->begin == rb->begin && ra->end == rb->end) {

        return 0;

    } else if (range_get_last(ra->begin, ra->end) <

               range_get_last(rb->begin, rb->end)) {

        return -1;

    } else {

        return 1;

    }

}
