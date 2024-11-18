void address_space_init_dispatch(AddressSpace *as)

{

    AddressSpaceDispatch *d = g_new(AddressSpaceDispatch, 1);



    d->phys_map  = (PhysPageEntry) { .ptr = PHYS_MAP_NODE_NIL, .is_leaf = 0 };

    d->listener = (MemoryListener) {

        .begin = mem_begin,

        .region_add = mem_add,

        .region_nop = mem_add,

        .priority = 0,

    };

    d->as = as;

    as->dispatch = d;

    memory_listener_register(&d->listener, as);

}
