static unsigned int * create_elf_tables(char *p, int argc, int envc,

                                        struct elfhdr * exec,

                                        unsigned long load_addr,

                                        unsigned long load_bias,

                                        unsigned long interp_load_addr, int ibcs,

                                        struct image_info *info)

{

        target_ulong *argv, *envp;

        target_ulong *sp, *csp;

        

	/*

	 * Force 16 byte _final_ alignment here for generality.

	 */

        sp = (unsigned int *) (~15UL & (unsigned long) p);

        csp = sp;

        csp -= (DLINFO_ITEMS + 1) * 2;

#ifdef DLINFO_ARCH_ITEMS

	csp -= DLINFO_ARCH_ITEMS*2;

#endif

        csp -= envc+1;

        csp -= argc+1;

	csp -= (!ibcs ? 3 : 1);	/* argc itself */

        if ((unsigned long)csp & 15UL)

            sp -= ((unsigned long)csp & 15UL) / sizeof(*sp);

        

#define NEW_AUX_ENT(nr, id, val) \

          put_user (tswapl(id), sp + (nr * 2)); \

          put_user (tswapl(val), sp + (nr * 2 + 1))

        sp -= 2;

        NEW_AUX_ENT (0, AT_NULL, 0);



	sp -= DLINFO_ITEMS*2;

        NEW_AUX_ENT( 0, AT_PHDR, (target_ulong)(load_addr + exec->e_phoff));

        NEW_AUX_ENT( 1, AT_PHENT, (target_ulong)(sizeof (struct elf_phdr)));

        NEW_AUX_ENT( 2, AT_PHNUM, (target_ulong)(exec->e_phnum));

        NEW_AUX_ENT( 3, AT_PAGESZ, (target_ulong)(TARGET_PAGE_SIZE));

        NEW_AUX_ENT( 4, AT_BASE, (target_ulong)(interp_load_addr));

        NEW_AUX_ENT( 5, AT_FLAGS, (target_ulong)0);

        NEW_AUX_ENT( 6, AT_ENTRY, load_bias + exec->e_entry);

        NEW_AUX_ENT( 7, AT_UID, (target_ulong) getuid());

        NEW_AUX_ENT( 8, AT_EUID, (target_ulong) geteuid());

        NEW_AUX_ENT( 9, AT_GID, (target_ulong) getgid());

        NEW_AUX_ENT(11, AT_EGID, (target_ulong) getegid());

#ifdef ARCH_DLINFO

	/* 

	 * ARCH_DLINFO must come last so platform specific code can enforce

	 * special alignment requirements on the AUXV if necessary (eg. PPC).

	 */

        ARCH_DLINFO;

#endif

#undef NEW_AUX_ENT



        sp -= envc+1;

        envp = sp;

        sp -= argc+1;

        argv = sp;

        if (!ibcs) {

                put_user(tswapl((target_ulong)envp),--sp);

                put_user(tswapl((target_ulong)argv),--sp);

        }

        put_user(tswapl(argc),--sp);

        info->arg_start = (unsigned int)((unsigned long)p & 0xffffffff);

        while (argc-->0) {

                put_user(tswapl((target_ulong)p),argv++);

                while (get_user(p++)) /* nothing */ ;

        }

        put_user(0,argv);

        info->arg_end = info->env_start = (unsigned int)((unsigned long)p & 0xffffffff);

        while (envc-->0) {

                put_user(tswapl((target_ulong)p),envp++);

                while (get_user(p++)) /* nothing */ ;

        }

        put_user(0,envp);

        info->env_end = (unsigned int)((unsigned long)p & 0xffffffff);

        return sp;

}
