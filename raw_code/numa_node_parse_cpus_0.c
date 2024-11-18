static void numa_node_parse_cpus(int nodenr, const char *cpus)

{

    char *endptr;

    unsigned long long value, endvalue;



    value = strtoull(cpus, &endptr, 10);

    if (*endptr == '-') {

        endvalue = strtoull(endptr+1, &endptr, 10);

    } else {

        endvalue = value;

    }



    if (!(endvalue < MAX_CPUMASK_BITS)) {

        endvalue = MAX_CPUMASK_BITS - 1;

        fprintf(stderr,

            "A max of %d CPUs are supported in a guest\n",

             MAX_CPUMASK_BITS);

    }



    bitmap_set(node_cpumask[nodenr], value, endvalue-value+1);

}
