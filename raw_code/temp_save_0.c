static inline void temp_save(TCGContext *s, TCGTemp *ts,

                             TCGRegSet allocated_regs)

{

#ifdef USE_LIVENESS_ANALYSIS

    /* ??? Liveness does not yet incorporate indirect bases.  */

    if (!ts->indirect_base) {

        /* The liveness analysis already ensures that globals are back

           in memory. Keep an assert for safety. */

        tcg_debug_assert(ts->val_type == TEMP_VAL_MEM || ts->fixed_reg);

        return;

    }

#endif

    temp_sync(s, ts, allocated_regs);

    temp_dead(s, ts);

}
