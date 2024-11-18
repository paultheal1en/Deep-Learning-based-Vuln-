static void render_memory_region(FlatView *view,

                                 MemoryRegion *mr,

                                 target_phys_addr_t base,

                                 AddrRange clip)

{

    MemoryRegion *subregion;

    unsigned i;

    target_phys_addr_t offset_in_region;

    uint64_t remain;

    uint64_t now;

    FlatRange fr;

    AddrRange tmp;



    base += mr->addr;



    tmp = addrrange_make(base, mr->size);



    if (!addrrange_intersects(tmp, clip)) {

        return;

    }



    clip = addrrange_intersection(tmp, clip);



    if (mr->alias) {

        base -= mr->alias->addr;

        base -= mr->alias_offset;

        render_memory_region(view, mr->alias, base, clip);

        return;

    }



    /* Render subregions in priority order. */

    QTAILQ_FOREACH(subregion, &mr->subregions, subregions_link) {

        render_memory_region(view, subregion, base, clip);

    }



    if (!mr->terminates) {

        return;

    }



    offset_in_region = clip.start - base;

    base = clip.start;

    remain = clip.size;



    /* Render the region itself into any gaps left by the current view. */

    for (i = 0; i < view->nr && remain; ++i) {

        if (base >= addrrange_end(view->ranges[i].addr)) {

            continue;

        }

        if (base < view->ranges[i].addr.start) {

            now = MIN(remain, view->ranges[i].addr.start - base);

            fr.mr = mr;

            fr.offset_in_region = offset_in_region;

            fr.addr = addrrange_make(base, now);

            fr.dirty_log_mask = mr->dirty_log_mask;

            flatview_insert(view, i, &fr);

            ++i;

            base += now;

            offset_in_region += now;

            remain -= now;

        }

        if (base == view->ranges[i].addr.start) {

            now = MIN(remain, view->ranges[i].addr.size);

            base += now;

            offset_in_region += now;

            remain -= now;

        }

    }

    if (remain) {

        fr.mr = mr;

        fr.offset_in_region = offset_in_region;

        fr.addr = addrrange_make(base, remain);

        fr.dirty_log_mask = mr->dirty_log_mask;

        flatview_insert(view, i, &fr);

    }

}
