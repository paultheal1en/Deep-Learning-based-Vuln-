static inline int alarm_has_dynticks(struct qemu_alarm_timer *t)

{

    return t && t->rearm;

}
