static void generate_bootsect(uint32_t gpr[8], uint16_t segs[6], uint16_t ip)

{

    uint8_t bootsect[512], *p;

    int i;

    int hda;



    hda = drive_get_index(IF_IDE, 0, 0);

    if (hda == -1) {

	fprintf(stderr, "A disk image must be given for 'hda' when booting "

		"a Linux kernel\n(if you really don't want it, use /dev/zero)\n");

	exit(1);

    }



    memset(bootsect, 0, sizeof(bootsect));



    /* Copy the MSDOS partition table if possible */

    bdrv_read(drives_table[hda].bdrv, 0, bootsect, 1);



    /* Make sure we have a partition signature */

    bootsect[510] = 0x55;

    bootsect[511] = 0xaa;



    /* Actual code */

    p = bootsect;

    *p++ = 0xfa;		/* CLI */

    *p++ = 0xfc;		/* CLD */



    for (i = 0; i < 6; i++) {

	if (i == 1)		/* Skip CS */

	    continue;



	*p++ = 0xb8;		/* MOV AX,imm16 */

	*p++ = segs[i];

	*p++ = segs[i] >> 8;

	*p++ = 0x8e;		/* MOV <seg>,AX */

	*p++ = 0xc0 + (i << 3);

    }



    for (i = 0; i < 8; i++) {

	*p++ = 0x66;		/* 32-bit operand size */

	*p++ = 0xb8 + i;	/* MOV <reg>,imm32 */

	*p++ = gpr[i];

	*p++ = gpr[i] >> 8;

	*p++ = gpr[i] >> 16;

	*p++ = gpr[i] >> 24;

    }



    *p++ = 0xea;		/* JMP FAR */

    *p++ = ip;			/* IP */

    *p++ = ip >> 8;

    *p++ = segs[1];		/* CS */

    *p++ = segs[1] >> 8;



    bdrv_set_boot_sector(drives_table[hda].bdrv, bootsect, sizeof(bootsect));

}
