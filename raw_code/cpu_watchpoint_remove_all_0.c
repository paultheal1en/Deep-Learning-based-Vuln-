void cpu_watchpoint_remove_all(CPUState *env, int mask)

{

    CPUWatchpoint *wp, *next;



    TAILQ_FOREACH_SAFE(wp, &env->watchpoints, entry, next) {

        if (wp->flags & mask)

            cpu_watchpoint_remove_by_ref(env, wp);

    }

}
