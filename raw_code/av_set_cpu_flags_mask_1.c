void av_set_cpu_flags_mask(int mask)

{

    cpu_flags = get_cpu_flags() & mask;

}
