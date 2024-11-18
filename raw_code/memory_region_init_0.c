void memory_region_init(MemoryRegion *mr,

                        const char *name,

                        uint64_t size)

{

    mr->ops = NULL;

    mr->parent = NULL;

    mr->size = int128_make64(size);

    if (size == UINT64_MAX) {

        mr->size = int128_2_64();

    }

    mr->addr = 0;

    mr->subpage = false;

    mr->enabled = true;

    mr->terminates = false;

    mr->ram = false;

    mr->romd_mode = true;

    mr->readonly = false;

    mr->rom_device = false;

    mr->destructor = memory_region_destructor_none;

    mr->priority = 0;

    mr->may_overlap = false;

    mr->alias = NULL;

    QTAILQ_INIT(&mr->subregions);

    memset(&mr->subregions_link, 0, sizeof mr->subregions_link);

    QTAILQ_INIT(&mr->coalesced);

    mr->name = g_strdup(name);

    mr->dirty_log_mask = 0;

    mr->ioeventfd_nb = 0;

    mr->ioeventfds = NULL;

    mr->flush_coalesced_mmio = false;

}
