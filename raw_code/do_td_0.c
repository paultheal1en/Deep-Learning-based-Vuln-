void do_td (int flags)

{

    if (!likely(!(((int64_t)T0 < (int64_t)T1 && (flags & 0x10)) ||

                  ((int64_t)T0 > (int64_t)T1 && (flags & 0x08)) ||

                  ((int64_t)T0 == (int64_t)T1 && (flags & 0x04)) ||

                  ((uint64_t)T0 < (uint64_t)T1 && (flags & 0x02)) ||

                  ((uint64_t)T0 > (uint64_t)T1 && (flags & 0x01)))))

        do_raise_exception_err(EXCP_PROGRAM, EXCP_TRAP);

}
