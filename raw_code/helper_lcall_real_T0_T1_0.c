void helper_lcall_real_T0_T1(int shift, int next_eip)

{

    int new_cs, new_eip;

    uint32_t esp, esp_mask;

    uint8_t *ssp;



    new_cs = T0;

    new_eip = T1;

    esp = env->regs[R_ESP];

    esp_mask = 0xffffffff;

    if (!(env->segs[R_SS].flags & DESC_B_MASK))

        esp_mask = 0xffff;

    ssp = env->segs[R_SS].base;

    if (shift) {

        esp -= 4;

        stl(ssp + (esp & esp_mask), env->segs[R_CS].selector);

        esp -= 4;

        stl(ssp + (esp & esp_mask), next_eip);

    } else {

        esp -= 2;

        stw(ssp + (esp & esp_mask), env->segs[R_CS].selector);

        esp -= 2;

        stw(ssp + (esp & esp_mask), next_eip);

    }



    if (!(env->segs[R_SS].flags & DESC_B_MASK))

        env->regs[R_ESP] = (env->regs[R_ESP] & ~0xffff) | (esp & 0xffff);

    else

        env->regs[R_ESP] = esp;

    env->eip = new_eip;

    env->segs[R_CS].selector = new_cs;

    env->segs[R_CS].base = (uint8_t *)(new_cs << 4);

}
