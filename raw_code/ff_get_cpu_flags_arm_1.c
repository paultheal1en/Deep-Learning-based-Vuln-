int ff_get_cpu_flags_arm(void)
{
    int flags = CORE_CPU_FLAGS;
    uint32_t hwcap;
    if (get_hwcap(&hwcap) < 0)
        if (get_cpuinfo(&hwcap) < 0)
            return flags;
#define check_cap(cap, flag) do {               \
        if (hwcap & HWCAP_ ## cap)              \
            flags |= AV_CPU_FLAG_ ## flag;      \
    } while (0)
    /* No flags explicitly indicate v6 or v6T2 so check others which
       imply support. */
    check_cap(EDSP,    ARMV5TE);
    check_cap(TLS,     ARMV6);
    check_cap(THUMBEE, ARMV6T2);
    check_cap(VFP,     VFP);
    check_cap(VFPv3,   VFPV3);
    check_cap(NEON,    NEON);
    /* The v6 checks above are not reliable so let higher flags
       trickle down. */
    if (flags & (AV_CPU_FLAG_VFPV3 | AV_CPU_FLAG_NEON))
        flags |= AV_CPU_FLAG_ARMV6T2;
    if (flags & AV_CPU_FLAG_ARMV6T2)
        flags |= AV_CPU_FLAG_ARMV6;
    return flags;
}