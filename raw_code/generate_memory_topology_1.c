static FlatView generate_memory_topology(MemoryRegion *mr)

{

    FlatView view;



    flatview_init(&view);



    render_memory_region(&view, mr, 0, addrrange_make(0, UINT64_MAX));

    flatview_simplify(&view);



    return view;

}
