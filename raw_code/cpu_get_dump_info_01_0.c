int cpu_get_dump_info(ArchDumpInfo *info,

                      const GuestPhysBlockList *guest_phys_blocks)

{

    bool lma = false;

    GuestPhysBlock *block;



#ifdef TARGET_X86_64

    X86CPU *first_x86_cpu = X86_CPU(first_cpu);



    lma = !!(first_x86_cpu->env.hflags & HF_LMA_MASK);

#endif



    if (lma) {

        info->d_machine = EM_X86_64;

    } else {

        info->d_machine = EM_386;

    }

    info->d_endian = ELFDATA2LSB;



    if (lma) {

        info->d_class = ELFCLASS64;

    } else {

        info->d_class = ELFCLASS32;



        QTAILQ_FOREACH(block, &guest_phys_blocks->head, next) {

            if (block->target_end > UINT_MAX) {

                /* The memory size is greater than 4G */

                info->d_class = ELFCLASS64;

                break;

            }

        }

    }



    return 0;

}
