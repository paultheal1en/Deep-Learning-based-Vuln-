static int pfpu_decode_insn(MilkymistPFPUState *s)

{

    uint32_t pc = s->regs[R_PC];

    uint32_t insn = s->microcode[pc];

    uint32_t reg_a = (insn >> 18) & 0x7f;

    uint32_t reg_b = (insn >> 11) & 0x7f;

    uint32_t op = (insn >> 7) & 0xf;

    uint32_t reg_d = insn & 0x7f;

    uint32_t r;

    int latency = 0;



    switch (op) {

    case OP_NOP:

        break;

    case OP_FADD:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = a + b;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_FADD;

        D_EXEC(qemu_log("ADD a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_FSUB:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = a - b;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_FSUB;

        D_EXEC(qemu_log("SUB a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_FMUL:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = a * b;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_FMUL;

        D_EXEC(qemu_log("MUL a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_FABS:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float t = fabsf(a);

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_FABS;

        D_EXEC(qemu_log("ABS a=%f t=%f, r=%08x\n", a, t, r));

    } break;

    case OP_F2I:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        int32_t t = a;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_F2I;

        D_EXEC(qemu_log("F2I a=%f t=%d, r=%08x\n", a, t, r));

    } break;

    case OP_I2F:

    {

        int32_t a = REINTERPRET_CAST(int32_t, s->gp_regs[reg_a]);

        float t = a;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_I2F;

        D_EXEC(qemu_log("I2F a=%08x t=%f, r=%08x\n", a, t, r));

    } break;

    case OP_VECTOUT:

    {

        uint32_t a = cpu_to_be32(s->gp_regs[reg_a]);

        uint32_t b = cpu_to_be32(s->gp_regs[reg_b]);

        target_phys_addr_t dma_ptr =

            get_dma_address(s->regs[R_MESHBASE],

                    s->gp_regs[GPR_X], s->gp_regs[GPR_Y]);

        cpu_physical_memory_write(dma_ptr, (uint8_t *)&a, 4);

        cpu_physical_memory_write(dma_ptr + 4, (uint8_t *)&b, 4);

        s->regs[R_LASTDMA] = dma_ptr + 4;

        D_EXEC(qemu_log("VECTOUT a=%08x b=%08x dma=%08x\n", a, b, dma_ptr));

        trace_milkymist_pfpu_vectout(a, b, dma_ptr);

    } break;

    case OP_SIN:

    {

        int32_t a = REINTERPRET_CAST(int32_t, s->gp_regs[reg_a]);

        float t = sinf(a * (1.0f / (M_PI * 4096.0f)));

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_SIN;

        D_EXEC(qemu_log("SIN a=%d t=%f, r=%08x\n", a, t, r));

    } break;

    case OP_COS:

    {

        int32_t a = REINTERPRET_CAST(int32_t, s->gp_regs[reg_a]);

        float t = cosf(a * (1.0f / (M_PI * 4096.0f)));

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_COS;

        D_EXEC(qemu_log("COS a=%d t=%f, r=%08x\n", a, t, r));

    } break;

    case OP_ABOVE:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = (a > b) ? 1.0f : 0.0f;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_ABOVE;

        D_EXEC(qemu_log("ABOVE a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_EQUAL:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = (a == b) ? 1.0f : 0.0f;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_EQUAL;

        D_EXEC(qemu_log("EQUAL a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_COPY:

    {

        r = s->gp_regs[reg_a];

        latency = LATENCY_COPY;

        D_EXEC(qemu_log("COPY"));

    } break;

    case OP_IF:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        uint32_t f = s->gp_regs[GPR_FLAGS];

        float t = (f != 0) ? a : b;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_IF;

        D_EXEC(qemu_log("IF f=%u a=%f b=%f t=%f, r=%08x\n", f, a, b, t, r));

    } break;

    case OP_TSIGN:

    {

        float a = REINTERPRET_CAST(float, s->gp_regs[reg_a]);

        float b = REINTERPRET_CAST(float, s->gp_regs[reg_b]);

        float t = (b < 0) ? -a : a;

        r = REINTERPRET_CAST(uint32_t, t);

        latency = LATENCY_TSIGN;

        D_EXEC(qemu_log("TSIGN a=%f b=%f t=%f, r=%08x\n", a, b, t, r));

    } break;

    case OP_QUAKE:

    {

        uint32_t a = s->gp_regs[reg_a];

        r = 0x5f3759df - (a >> 1);

        latency = LATENCY_QUAKE;

        D_EXEC(qemu_log("QUAKE a=%d r=%08x\n", a, r));

    } break;



    default:

        error_report("milkymist_pfpu: unknown opcode %d\n", op);

        break;

    }



    if (!reg_d) {

        D_EXEC(qemu_log("%04d %8s R%03d, R%03d <L=%d, E=%04d>\n",

                    s->regs[R_PC], opcode_to_str[op], reg_a, reg_b, latency,

                    s->regs[R_PC] + latency));

    } else {

        D_EXEC(qemu_log("%04d %8s R%03d, R%03d <L=%d, E=%04d> -> R%03d\n",

                    s->regs[R_PC], opcode_to_str[op], reg_a, reg_b, latency,

                    s->regs[R_PC] + latency, reg_d));

    }



    if (op == OP_VECTOUT) {

        return 0;

    }



    /* store output for this cycle */

    if (reg_d) {

        uint32_t val = output_queue_remove(s);

        D_EXEC(qemu_log("R%03d <- 0x%08x\n", reg_d, val));

        s->gp_regs[reg_d] = val;

    }



    output_queue_advance(s);



    /* store op output */

    if (op != OP_NOP) {

        output_queue_insert(s, r, latency-1);

    }



    /* advance PC */

    s->regs[R_PC]++;



    return 1;

};
