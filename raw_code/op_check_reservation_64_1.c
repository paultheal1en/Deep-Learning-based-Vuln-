void OPPROTO op_check_reservation_64 (void)

{

    if ((uint64_t)env->reserve == (uint64_t)(T0 & ~0x00000003))

        env->reserve = -1;

    RETURN();

}
