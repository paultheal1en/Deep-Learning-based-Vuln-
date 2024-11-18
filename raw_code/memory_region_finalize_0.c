static void memory_region_finalize(Object *obj)

{

    MemoryRegion *mr = MEMORY_REGION(obj);



    assert(QTAILQ_EMPTY(&mr->subregions));

    assert(memory_region_transaction_depth == 0);

    mr->destructor(mr);

    memory_region_clear_coalescing(mr);

    g_free((char *)mr->name);

    g_free(mr->ioeventfds);

}
