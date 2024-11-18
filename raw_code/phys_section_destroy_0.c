static void phys_section_destroy(MemoryRegion *mr)

{

    memory_region_unref(mr);



    if (mr->subpage) {

        subpage_t *subpage = container_of(mr, subpage_t, iomem);

        object_unref(OBJECT(&subpage->iomem));

        g_free(subpage);

    }

}
