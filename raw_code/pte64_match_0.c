static bool pte64_match(target_ulong pte0, target_ulong pte1,

                        bool secondary, target_ulong ptem)

{

    return (pte0 & HPTE64_V_VALID)

        && (secondary == !!(pte0 & HPTE64_V_SECONDARY))

        && HPTE64_V_COMPARE(pte0, ptem);

}
