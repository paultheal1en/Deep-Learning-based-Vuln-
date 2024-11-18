static void mem_commit(MemoryListener *listener)

{

    AddressSpace *as = container_of(listener, AddressSpace, dispatch_listener);

    AddressSpaceDispatch *cur = as->dispatch;

    AddressSpaceDispatch *next = as->next_dispatch;



    next->nodes = next_map.nodes;

    next->sections = next_map.sections;



    phys_page_compact_all(next, next_map.nodes_nb);



    as->dispatch = next;

    g_free(cur);

}
