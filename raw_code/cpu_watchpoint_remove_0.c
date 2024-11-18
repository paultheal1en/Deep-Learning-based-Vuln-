int cpu_watchpoint_remove(CPUState *env, target_ulong addr, target_ulong len,

                          int flags)

{

    target_ulong len_mask = ~(len - 1);

    CPUWatchpoint *wp;



    TAILQ_FOREACH(wp, &env->watchpoints, entry) {

        if (addr == wp->vaddr && len_mask == wp->len_mask

                && flags == (wp->flags & ~BP_WATCHPOINT_HIT)) {

            cpu_watchpoint_remove_by_ref(env, wp);

            return 0;

        }

    }

    return -ENOENT;

}
