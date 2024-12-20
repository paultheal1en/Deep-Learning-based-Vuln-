static void vga_update_memory_access(VGACommonState *s)

{

    hwaddr base, offset, size;



    if (s->legacy_address_space == NULL) {

        return;

    }



    if (s->has_chain4_alias) {

        memory_region_del_subregion(s->legacy_address_space, &s->chain4_alias);

        object_unparent(OBJECT(&s->chain4_alias));

        s->has_chain4_alias = false;

        s->plane_updated = 0xf;

    }

    if ((s->sr[VGA_SEQ_PLANE_WRITE] & VGA_SR02_ALL_PLANES) ==

        VGA_SR02_ALL_PLANES && s->sr[VGA_SEQ_MEMORY_MODE] & VGA_SR04_CHN_4M) {

        offset = 0;

        switch ((s->gr[VGA_GFX_MISC] >> 2) & 3) {

        case 0:

            base = 0xa0000;

            size = 0x20000;

            break;

        case 1:

            base = 0xa0000;

            size = 0x10000;

            offset = s->bank_offset;

            break;

        case 2:

            base = 0xb0000;

            size = 0x8000;

            break;

        case 3:

        default:

            base = 0xb8000;

            size = 0x8000;

            break;

        }

        assert(offset + size <= s->vram_size);

        memory_region_init_alias(&s->chain4_alias, memory_region_owner(&s->vram),

                                 "vga.chain4", &s->vram, offset, size);

        memory_region_add_subregion_overlap(s->legacy_address_space, base,

                                            &s->chain4_alias, 2);

        s->has_chain4_alias = true;

    }

}
