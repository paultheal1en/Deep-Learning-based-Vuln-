void do_tw (int flags)

{

    if (!likely(!(((int32_t)T0 < (int32_t)T1 && (flags & 0x10)) ||

                  ((int32_t)T0 > (int32_t)T1 && (flags & 0x08)) ||

                  ((int32_t)T0 == (int32_t)T1 && (flags & 0x04)) ||

                  ((uint32_t)T0 < (uint32_t)T1 && (flags & 0x02)) ||

                  ((uint32_t)T0 > (uint32_t)T1 && (flags & 0x01))))) {

        do_raise_exception_err(EXCP_PROGRAM, EXCP_TRAP);

    }

}
