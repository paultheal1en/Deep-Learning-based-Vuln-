static inline int get_lowest_part_list_y(H264Context *h, Picture *pic, int n,

                                         int height, int y_offset, int list)

{

    int raw_my        = h->mv_cache[list][scan8[n]][1];

    int filter_height = (raw_my & 3) ? 2 : 0;

    int full_my       = (raw_my >> 2) + y_offset;

    int top           = full_my - filter_height;

    int bottom        = full_my + filter_height + height;



    return FFMAX(abs(top), bottom);

}
