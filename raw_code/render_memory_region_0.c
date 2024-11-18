static void render_memory_region(FlatView *view,

                                 MemoryRegion *mr,

                                 Int128 base,

                                 AddrRange clip,

                                 bool readonly)

{

    MemoryRegion *subregion;

    unsigned i;

    hwaddr offset_in_region;

    Int128 remain;

    Int128 now;

    FlatRange fr;

    AddrRange tmp;



    if (!mr->enabled) {

        return;

    }



    int128_addto(&base, int128_make64(mr->addr));

    readonly |= mr->readonly;



    tmp = addrrange_make(base, mr->size);



    if (!addrrange_intersects(tmp, clip)) {

        return;

    }



    clip = addrrange_intersection(tmp, clip);



    if (mr->alias) {

        int128_subfrom(&base, int128_make64(mr->alias->addr));

        int128_subfrom(&base, int128_make64(mr->alias_offset));

        render_memory_region(view, mr->alias, base, clip, readonly);

        return;

    }



    /* Render subregions in priority order. */

    QTAILQ_FOREACH(subregion, &mr->subregions, subregions_link) {

        render_memory_region(view, subregion, base, clip, readonly);

    }



    if (!mr->terminates) {

        return;

    }



    offset_in_region = int128_get64(int128_sub(clip.start, base));

    base = clip.start;

    remain = clip.size;



    /* Render the region itself into any gaps left by the current view. */

    for (i = 0; i < view->nr && int128_nz(remain); ++i) {

        if (int128_ge(base, addrrange_end(view->ranges[i].addr))) {

            continue;

        }

        if (int128_lt(base, view->ranges[i].addr.start)) {

            now = int128_min(remain,

                             int128_sub(view->ranges[i].addr.start, base));

            fr.mr = mr;

            fr.offset_in_region = offset_in_region;

            fr.addr = addrrange_make(base, now);

            fr.dirty_log_mask = mr->dirty_log_mask;

            fr.romd_mode = mr->romd_mode;

            fr.readonly = readonly;

            flatview_insert(view, i, &fr);

            ++i;

            int128_addto(&base, now);

            offset_in_region += int128_get64(now);

            int128_subfrom(&remain, now);

        }

        now = int128_sub(int128_min(int128_add(base, remain),

                                    addrrange_end(view->ranges[i].addr)),

                         base);

        int128_addto(&base, now);

        offset_in_region += int128_get64(now);

        int128_subfrom(&remain, now);

    }

    if (int128_nz(remain)) {

        fr.mr = mr;

        fr.offset_in_region = offset_in_region;

        fr.addr = addrrange_make(base, remain);

        fr.dirty_log_mask = mr->dirty_log_mask;

        fr.romd_mode = mr->romd_mode;

        fr.readonly = readonly;

        flatview_insert(view, i, &fr);

    }

}
