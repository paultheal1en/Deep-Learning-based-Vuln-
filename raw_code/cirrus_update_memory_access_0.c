static void cirrus_update_memory_access(CirrusVGAState *s)

{

    unsigned mode;



    if ((s->sr[0x17] & 0x44) == 0x44) {

        goto generic_io;

    } else if (s->cirrus_srcptr != s->cirrus_srcptr_end) {

        goto generic_io;

    } else {

	if ((s->gr[0x0B] & 0x14) == 0x14) {

            goto generic_io;

	} else if (s->gr[0x0B] & 0x02) {

            goto generic_io;

        }



	mode = s->gr[0x05] & 0x7;

	if (mode < 4 || mode > 5 || ((s->gr[0x0B] & 0x4) == 0)) {

            map_linear_vram(s);

            s->cirrus_linear_write[0] = cirrus_linear_mem_writeb;

            s->cirrus_linear_write[1] = cirrus_linear_mem_writew;

            s->cirrus_linear_write[2] = cirrus_linear_mem_writel;

        } else {

        generic_io:

            unmap_linear_vram(s);

            s->cirrus_linear_write[0] = cirrus_linear_writeb;

            s->cirrus_linear_write[1] = cirrus_linear_writew;

            s->cirrus_linear_write[2] = cirrus_linear_writel;

        }

    }

}
