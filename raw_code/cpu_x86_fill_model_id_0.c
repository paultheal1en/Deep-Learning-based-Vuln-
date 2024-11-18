static int cpu_x86_fill_model_id(char *str)

{

    uint32_t eax, ebx, ecx, edx;

    int i;



    for (i = 0; i < 3; i++) {

        host_cpuid(0x80000002 + i, 0, &eax, &ebx, &ecx, &edx);

        memcpy(str + i * 16 +  0, &eax, 4);

        memcpy(str + i * 16 +  4, &ebx, 4);

        memcpy(str + i * 16 +  8, &ecx, 4);

        memcpy(str + i * 16 + 12, &edx, 4);

    }

    return 0;

}
