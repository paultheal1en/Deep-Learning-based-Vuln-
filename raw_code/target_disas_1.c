void target_disas(FILE *out, CPUState *cpu, target_ulong code,

                  target_ulong size, int flags)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

    target_ulong pc;

    int count;

    CPUDebug s;



    INIT_DISASSEMBLE_INFO(s.info, out, fprintf);



    s.cpu = cpu;

    s.info.read_memory_func = target_read_memory;

    s.info.read_memory_inner_func = NULL;

    s.info.buffer_vma = code;

    s.info.buffer_length = size;

    s.info.print_address_func = generic_print_address;



#ifdef TARGET_WORDS_BIGENDIAN

    s.info.endian = BFD_ENDIAN_BIG;

#else

    s.info.endian = BFD_ENDIAN_LITTLE;

#endif



    if (cc->disas_set_info) {

        cc->disas_set_info(cpu, &s.info);

    }



#if defined(TARGET_I386)

    if (flags == 2) {

        s.info.mach = bfd_mach_x86_64;

    } else if (flags == 1) {

        s.info.mach = bfd_mach_i386_i8086;

    } else {

        s.info.mach = bfd_mach_i386_i386;

    }

    s.info.print_insn = print_insn_i386;

#elif defined(TARGET_PPC)

    if ((flags >> 16) & 1) {

        s.info.endian = BFD_ENDIAN_LITTLE;

    }

    if (flags & 0xFFFF) {

        /* If we have a precise definition of the instruction set, use it. */

        s.info.mach = flags & 0xFFFF;

    } else {

#ifdef TARGET_PPC64

        s.info.mach = bfd_mach_ppc64;

#else

        s.info.mach = bfd_mach_ppc;

#endif

    }

    s.info.disassembler_options = (char *)"any";

    s.info.print_insn = print_insn_ppc;

#endif

    if (s.info.print_insn == NULL) {

        s.info.print_insn = print_insn_od_target;

    }



    for (pc = code; size > 0; pc += count, size -= count) {

	fprintf(out, "0x" TARGET_FMT_lx ":  ", pc);

	count = s.info.print_insn(pc, &s.info);

#if 0

        {

            int i;

            uint8_t b;

            fprintf(out, " {");

            for(i = 0; i < count; i++) {

                target_read_memory(pc + i, &b, 1, &s.info);

                fprintf(out, " %02x", b);

            }

            fprintf(out, " }");

        }

#endif

	fprintf(out, "\n");

	if (count < 0)

	    break;

        if (size < count) {

            fprintf(out,

                    "Disassembler disagrees with translator over instruction "

                    "decoding\n"

                    "Please report this to qemu-devel@nongnu.org\n");

            break;

        }

    }

}
