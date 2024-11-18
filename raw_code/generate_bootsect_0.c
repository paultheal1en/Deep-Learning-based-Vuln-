static void generate_bootsect(target_phys_addr_t option_rom,

                              uint32_t gpr[8], uint16_t segs[6], uint16_t ip)

{

    uint8_t rom[512], *p, *reloc;

    uint8_t sum;

    int i;



    memset(rom, 0, sizeof(rom));



    p = rom;

    /* Make sure we have an option rom signature */

    *p++ = 0x55;

    *p++ = 0xaa;



    /* ROM size in sectors*/

    *p++ = 1;



    /* Hook int19 */



    *p++ = 0x50;		/* push ax */

    *p++ = 0x1e;		/* push ds */

    *p++ = 0x31; *p++ = 0xc0;	/* xor ax, ax */

    *p++ = 0x8e; *p++ = 0xd8;	/* mov ax, ds */



    *p++ = 0xc7; *p++ = 0x06;   /* movvw _start,0x64 */

    *p++ = 0x64; *p++ = 0x00;

    reloc = p;

    *p++ = 0x00; *p++ = 0x00;



    *p++ = 0x8c; *p++ = 0x0e;   /* mov cs,0x66 */

    *p++ = 0x66; *p++ = 0x00;



    *p++ = 0x1f;		/* pop ds */

    *p++ = 0x58;		/* pop ax */

    *p++ = 0xcb;		/* lret */



    /* Actual code */

    *reloc = (p - rom);



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



    /* sign rom */

    sum = 0;

    for (i = 0; i < (sizeof(rom) - 1); i++)

        sum += rom[i];

    rom[sizeof(rom) - 1] = -sum;



    cpu_physical_memory_write_rom(option_rom, rom, sizeof(rom));

    option_rom_setup_reset(option_rom, sizeof (rom));

}
