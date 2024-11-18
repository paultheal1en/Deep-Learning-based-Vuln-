void powerpc_display_perf_report(void)

{

  int i;

#ifndef POWERPC_PERF_USE_PMC

  fprintf(stderr, "PowerPC performance report\n Values are from the Time Base register, and represent 4 bus cycles.\n");

#else /* POWERPC_PERF_USE_PMC */

  fprintf(stderr, "PowerPC performance report\n Values are from the PMC registers, and represent whatever the registers are set to record.\n");

#endif /* POWERPC_PERF_USE_PMC */

  for(i = 0 ; i < powerpc_perf_total ; i++)

  {

    if (perfdata[i][powerpc_data_num] != (unsigned long long)0)

      fprintf(stderr, " Function \"%s\" (pmc1):\n\tmin: %llu\n\tmax: %llu\n\tavg: %1.2lf (%llu)\n",

              perfname[i],

              perfdata[i][powerpc_data_min],

              perfdata[i][powerpc_data_max],

              (double)perfdata[i][powerpc_data_sum] /

              (double)perfdata[i][powerpc_data_num],

              perfdata[i][powerpc_data_num]);

#ifdef POWERPC_PERF_USE_PMC

    if (perfdata_pmc2[i][powerpc_data_num] != (unsigned long long)0)

      fprintf(stderr, " Function \"%s\" (pmc2):\n\tmin: %llu\n\tmax: %llu\n\tavg: %1.2lf (%llu)\n",

              perfname[i],

              perfdata_pmc2[i][powerpc_data_min],

              perfdata_pmc2[i][powerpc_data_max],

              (double)perfdata_pmc2[i][powerpc_data_sum] /

              (double)perfdata_pmc2[i][powerpc_data_num],

              perfdata_pmc2[i][powerpc_data_num]);

    if (perfdata_pmc3[i][powerpc_data_num] != (unsigned long long)0)

      fprintf(stderr, " Function \"%s\" (pmc3):\n\tmin: %llu\n\tmax: %llu\n\tavg: %1.2lf (%llu)\n",

              perfname[i],

              perfdata_pmc3[i][powerpc_data_min],

              perfdata_pmc3[i][powerpc_data_max],

              (double)perfdata_pmc3[i][powerpc_data_sum] /

              (double)perfdata_pmc3[i][powerpc_data_num],

              perfdata_pmc3[i][powerpc_data_num]);

#endif

  }

}
