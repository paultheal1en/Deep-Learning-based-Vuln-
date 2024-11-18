static void sys_write(void *opaque, target_phys_addr_t addr,

                      uint64_t value, unsigned size)

{

    LM32SysState *s = opaque;

    char *testname;



    trace_lm32_sys_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_CTRL:

        qemu_system_shutdown_request();

        break;

    case R_PASSFAIL:

        s->regs[addr] = value;

        testname = (char *)s->testname;

        qemu_log("TC  %-16s %s\n", testname, (value) ? "FAILED" : "OK");

        break;

    case R_TESTNAME:

        s->regs[addr] = value;

        copy_testname(s);

        break;



    default:

        error_report("lm32_sys: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
