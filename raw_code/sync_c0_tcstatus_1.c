static void sync_c0_tcstatus(CPUMIPSState *cpu, int tc,

                             target_ulong v)

{

    uint32_t status;

    uint32_t tcu, tmx, tasid, tksu;

    uint32_t mask = ((1 << CP0St_CU3)

                       | (1 << CP0St_CU2)

                       | (1 << CP0St_CU1)

                       | (1 << CP0St_CU0)

                       | (1 << CP0St_MX)

                       | (3 << CP0St_KSU));



    tcu = (v >> CP0TCSt_TCU0) & 0xf;

    tmx = (v >> CP0TCSt_TMX) & 0x1;

    tasid = v & 0xff;

    tksu = (v >> CP0TCSt_TKSU) & 0x3;



    status = tcu << CP0St_CU0;

    status |= tmx << CP0St_MX;

    status |= tksu << CP0St_KSU;



    cpu->CP0_Status &= ~mask;

    cpu->CP0_Status |= status;



    /* Sync the TASID with EntryHi.  */

    cpu->CP0_EntryHi &= ~0xff;

    cpu->CP0_EntryHi = tasid;



    compute_hflags(cpu);

}
