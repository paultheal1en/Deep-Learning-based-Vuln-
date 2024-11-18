static void map_linear_vram(CirrusVGAState *s)

{




    if (!s->map_addr && s->lfb_addr && s->lfb_end) {

        s->map_addr = s->lfb_addr;

        s->map_end = s->lfb_end;

        cpu_register_physical_memory(s->map_addr, s->map_end - s->map_addr, s->vram_offset);

    }



    if (!s->map_addr)

        return;



    s->lfb_vram_mapped = 0;







    if (!(s->cirrus_srcptr != s->cirrus_srcptr_end)

        && !((s->sr[0x07] & 0x01) == 0)

        && !((s->gr[0x0B] & 0x14) == 0x14)

        && !(s->gr[0x0B] & 0x02)) {





                                    (s->vram_offset + s->cirrus_bank_base[0]) | IO_MEM_RAM);


                                    (s->vram_offset + s->cirrus_bank_base[1]) | IO_MEM_RAM);



        s->lfb_vram_mapped = 1;

    }

    else {

        cpu_register_physical_memory(isa_mem_base + 0xa0000, 0x20000,

                                     s->vga_io_memory);

    }



    vga_dirty_log_start((VGAState *)s);

}