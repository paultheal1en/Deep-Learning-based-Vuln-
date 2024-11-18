static void rc4030_dma_as_update_one(rc4030State *s, int index, uint32_t frame)

{

    if (index < MAX_TL_ENTRIES) {

        memory_region_set_enabled(&s->dma_mrs[index], false);

    }



    if (!frame) {

        return;

    }



    if (index >= MAX_TL_ENTRIES) {

        qemu_log_mask(LOG_UNIMP,

                      "rc4030: trying to use too high "

                      "translation table entry %d (max allowed=%d)",

                      index, MAX_TL_ENTRIES);

        return;

    }

    memory_region_set_alias_offset(&s->dma_mrs[index], frame);

    memory_region_set_enabled(&s->dma_mrs[index], true);

}
