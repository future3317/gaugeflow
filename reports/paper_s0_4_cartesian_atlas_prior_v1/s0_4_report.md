# S0.4 Cartesian Atlas Prior qualification

Decision: **failed_no_advance**.

This protocol qualifies the atlas as a replacement prior and quotient interface. It does not require reproduction of the archived Hopf posterior.

Failed primary checks: ['cuda_latency'].

```json
{
  "protocol_id": "paper_s0_4_cartesian_atlas_prior_v1",
  "decision": "failed_no_advance",
  "checks": {
    "cpu_representative_tensor": true,
    "cpu_representative_token": true,
    "cpu_representative_posterior": true,
    "cpu_representative_response": true,
    "cuda_representative_tensor": true,
    "cuda_representative_token": true,
    "cuda_representative_posterior": true,
    "cuda_representative_response": true,
    "proper_rotation": true,
    "polar_parity": true,
    "stratum_boundary": true,
    "stratum_infinitesimal_continuity": true,
    "stratum_backward_gradients": true,
    "candidate_enumeration_order": true,
    "candidate_enumeration_order_posterior": true,
    "candidate_duplicate_expansion": true,
    "candidate_duplicate_expansion_posterior": true,
    "generic_candidate_count": true,
    "nonzero_descriptor_isotropic": true,
    "axial_refinement": true,
    "synthetic_coverage": true,
    "fp32_aligned_reference": true,
    "fp32_response_reference": true,
    "fp32_token_reference": true,
    "fp32_posterior_reference": true,
    "bf16_aligned_reference": true,
    "bf16_response_reference": true,
    "bf16_token_reference": true,
    "bf16_posterior_reference": true,
    "bf16_finite": true,
    "translation": true,
    "unimodular": true,
    "zero_null": true,
    "cuda_latency": false,
    "cuda_memory": true,
    "finite": true
  },
  "cpu_unseen_panel": [
    {
      "representative_posterior_l1_error": 3.016126834317354e-09,
      "candidate_pushforward_max_error": 3.2032573113777515e-08,
      "representative_tensor_error": 6.28071517956991e-09,
      "representative_response_relative_error": 6.355344809432152e-08,
      "representative_token_error": 0.00012886577518771806,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.0419410632303994e-08,
      "candidate_pushforward_max_error": 8.245857474011288e-08,
      "representative_tensor_error": 8.3453252610439e-09,
      "representative_response_relative_error": 1.1798518760895836e-07,
      "representative_token_error": 2.957984003362076e-05,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 5.264043564966788e-08,
      "candidate_pushforward_max_error": 2.1853900845424632e-07,
      "representative_tensor_error": 3.790632240446086e-08,
      "representative_response_relative_error": 1.139583128190376e-07,
      "representative_token_error": 0.0010846713206638149,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.9272844552643397e-09,
      "candidate_pushforward_max_error": 1.5697602357148138e-08,
      "representative_tensor_error": 3.2752010344584616e-09,
      "representative_response_relative_error": 8.499884302117681e-08,
      "representative_token_error": 1.4765167142246291e-06,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 5.2897201217622525e-08,
      "candidate_pushforward_max_error": 2.6578564580034045e-07,
      "representative_tensor_error": 1.8350698860338227e-08,
      "representative_response_relative_error": 4.0673767021215005e-08,
      "representative_token_error": 0.0014774990538565524,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 4.948115691678521e-09,
      "candidate_pushforward_max_error": 2.9182221474540697e-08,
      "representative_tensor_error": 1.4256762843007784e-08,
      "representative_response_relative_error": 1.0403482028446098e-07,
      "representative_token_error": 0.00011191390659206132,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 3.7437246687844605e-09,
      "candidate_pushforward_max_error": 2.98655273821735e-08,
      "representative_tensor_error": 5.749103403249043e-09,
      "representative_response_relative_error": 1.147518116574116e-07,
      "representative_token_error": 2.504903298267202e-05,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 9.266391225816245e-09,
      "candidate_pushforward_max_error": 3.461798906137251e-08,
      "representative_tensor_error": 3.149324453757072e-08,
      "representative_response_relative_error": 1.4154666585098962e-07,
      "representative_token_error": 0.0002892440038298591,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    }
  ],
  "cuda_unseen_panel": [
    {
      "representative_posterior_l1_error": 1.4023680705577135e-07,
      "candidate_pushforward_max_error": 1.6739782040531281e-06,
      "representative_tensor_error": 4.21425113472651e-07,
      "representative_response_relative_error": 4.620070740202209e-06,
      "representative_token_error": 4.850881305173971e-05,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 5.640322342514992e-08,
      "candidate_pushforward_max_error": 1.359542125101143e-06,
      "representative_tensor_error": 5.7158857202921354e-08,
      "representative_response_relative_error": 2.622211241032346e-06,
      "representative_token_error": 9.315253009845037e-06,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.595763023942709e-07,
      "candidate_pushforward_max_error": 2.1983128135616425e-06,
      "representative_tensor_error": 1.3596870473975287e-07,
      "representative_response_relative_error": 1.7861509604699677e-06,
      "representative_token_error": 0.0008656876161694527,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.6577541828155518e-07,
      "candidate_pushforward_max_error": 7.108963018254144e-07,
      "representative_tensor_error": 6.47319666313706e-07,
      "representative_response_relative_error": 1.998557308979798e-06,
      "representative_token_error": 0.00048711622366681695,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.7462298274040222e-07,
      "candidate_pushforward_max_error": 6.828976211181725e-07,
      "representative_tensor_error": 1.5705630573847884e-07,
      "representative_response_relative_error": 6.939298600627808e-06,
      "representative_token_error": 6.975213182158768e-05,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.1918018572032452e-07,
      "candidate_pushforward_max_error": 8.153876933647553e-07,
      "representative_tensor_error": 6.910021710382352e-08,
      "representative_response_relative_error": 4.161428535098821e-07,
      "representative_token_error": 3.0079809221206233e-05,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 7.060589268803596e-08,
      "candidate_pushforward_max_error": 1.4289001910583465e-06,
      "representative_tensor_error": 1.7150640019281127e-07,
      "representative_response_relative_error": 4.3861423364432994e-06,
      "representative_token_error": 6.4999130700016394e-06,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    },
    {
      "representative_posterior_l1_error": 1.7872662283480167e-07,
      "candidate_pushforward_max_error": 9.298611303165671e-07,
      "representative_tensor_error": 5.013714599044761e-07,
      "representative_response_relative_error": 1.8767995868529397e-07,
      "representative_token_error": 0.0008148152846843004,
      "generic_raw_candidate_count": 4032,
      "generic_unique_candidate_count": 4032
    }
  ],
  "parity_and_zero_null": {
    "candidate_count": 4032,
    "proper_rotation_determinant_error": 2.0119952148789366e-08,
    "polar_rank3_parity_error": 0.0,
    "physical_zero_null_token_distance": 0.7545973914670261,
    "nonzero_descriptor_isotropic_unique_candidates": 4032
  },
  "candidate_measure": {
    "generic_raw_count": 4032,
    "generic_unique_count": 4032,
    "generic_multiplicity_min_max": [
      1,
      1
    ],
    "generic_effective_prior_rank": 4032,
    "one_sided_axial_raw_count": 32256,
    "one_sided_axial_unique_count": 8064,
    "worst_axial_raw_count": 258048,
    "worst_axial_unique_count": 16128,
    "enumeration_order_posterior_difference": 0.0,
    "enumeration_order_aligned_difference": 0.0,
    "duplicate_expansion_posterior_difference": 0.0,
    "duplicate_expansion_aligned_difference": 0.0
  },
  "stratum_boundary": {
    "gap_multipliers": [
      0.2,
      0.49,
      0.5,
      0.8,
      0.999,
      1.001,
      1.2,
      1.99,
      2.0,
      2.01,
      3.0
    ],
    "raw_candidate_counts": [
      32256,
      32256,
      32256,
      36288,
      36288,
      36288,
      36288,
      36288,
      36288,
      4032,
      4032
    ],
    "unique_candidate_counts": [
      8064,
      8064,
      8064,
      8064,
      8064,
      8064,
      8064,
      8064,
      8064,
      4032,
      4032
    ],
    "normalized_jumps": [
      0.0,
      0.0,
      0.006062179422168016,
      0.00863315645069203,
      9.698418980496305e-05,
      0.010058284088294215,
      0.026086762314162678,
      5.8156478488904135e-06,
      9.336419624257868e-13,
      0.0
    ],
    "maximum_normalized_jump": 0.026086762314162678,
    "infinitesimal_normalized_jumps": [
      7.894120062053484e-10,
      9.698426533980966e-06,
      6.312832308411326e-10
    ],
    "maximum_infinitesimal_normalized_jump": 9.698426533980966e-06,
    "all_backward_gradients_finite": true
  },
  "axial_refinement": {
    "circle_samples": [
      8,
      16,
      32,
      64
    ],
    "raw_candidate_counts": [
      32256,
      64512,
      129024,
      258048
    ],
    "unique_candidate_counts": [
      8064,
      16128,
      32256,
      64512
    ],
    "successive_normalized_differences": [
      0.03596496966445884,
      0.005285223303377605,
      3.9071613676857365e-06
    ],
    "successive_differences_monotone": true
  },
  "synthetic_coverage": {
    "panel": [
      {
        "nearest_geodesic": 0.08987794130741886,
        "posterior_mode_geodesic": 0.1328604950607754,
        "nearest_candidate_is_posterior_mode": false,
        "nearest_candidate_posterior_mass": 0.22993861044085323
      },
      {
        "nearest_geodesic": 0.24777715418795396,
        "posterior_mode_geodesic": 0.38066657815319516,
        "nearest_candidate_is_posterior_mode": false,
        "nearest_candidate_posterior_mass": 0.05613746163712869
      },
      {
        "nearest_geodesic": 0.16033978150904707,
        "posterior_mode_geodesic": 0.16033978150904707,
        "nearest_candidate_is_posterior_mode": true,
        "nearest_candidate_posterior_mass": 0.3959918160342114
      },
      {
        "nearest_geodesic": 0.12016887249118664,
        "posterior_mode_geodesic": 0.12016887249118664,
        "nearest_candidate_is_posterior_mode": true,
        "nearest_candidate_posterior_mass": 0.28718453286878864
      },
      {
        "nearest_geodesic": 0.13439504957038464,
        "posterior_mode_geodesic": 0.13439504957038464,
        "nearest_candidate_is_posterior_mode": true,
        "nearest_candidate_posterior_mass": 0.3147807179210093
      },
      {
        "nearest_geodesic": 0.24017231725569863,
        "posterior_mode_geodesic": 0.3019211980772202,
        "nearest_candidate_is_posterior_mode": false,
        "nearest_candidate_posterior_mass": 0.09238739375838707
      },
      {
        "nearest_geodesic": 0.09424434053833637,
        "posterior_mode_geodesic": 0.09424434053833637,
        "nearest_candidate_is_posterior_mode": true,
        "nearest_candidate_posterior_mass": 0.3113308356009715
      },
      {
        "nearest_geodesic": 0.08580300087116195,
        "posterior_mode_geodesic": 0.08580300087116195,
        "nearest_candidate_is_posterior_mode": true,
        "nearest_candidate_posterior_mass": 0.36237751699904797
      }
    ],
    "maximum_nearest_geodesic": 0.24777715418795396,
    "mean_posterior_mode_geodesic": 0.17629991453391342,
    "posterior_mode_retrieval_rate": 0.625
  },
  "mixed_precision_reference": {
    "status": "NVIDIA GeForce RTX 4060 Ti",
    "fp32_vs_fp64": {
      "aligned_relative_error": 4.6825964942835704e-07,
      "response_relative_error": 3.9018773737901256e-05,
      "token_relative_error": 3.6136272724704774e-07,
      "sorted_posterior_l1_error": 2.439788101914588e-07,
      "unique_candidate_count": 4032
    },
    "bf16_autocast_vs_fp64": {
      "aligned_relative_error": 0.007806546302569102,
      "response_relative_error": 0.013610656626271539,
      "token_relative_error": 0.010545308669326186,
      "sorted_posterior_l1_error": 0.0006394597905191096,
      "unique_candidate_count": 4032
    },
    "finite": true
  },
  "denoiser": {
    "translation_max_error": 6.556510925292969e-07,
    "unimodular_max_error": 5.960464477539062e-07
  },
  "cuda_benchmark": {
    "status": "NVIDIA GeForce RTX 4060 Ti",
    "atlas_ms_per_forward": 41.88768277000008,
    "atlas_peak_memory_mb": 15.4208984375,
    "archived_hopf_k3840_ms_per_forward": 109.23637200000087,
    "archived_hopf_k3840_peak_memory_mb": 13.93603515625,
    "finite": true
  },
  "hopf_comparison_role": "diagnostic_only_not_an_acceptance_check",
  "runtime": {
    "torch": "2.5.1+cu124",
    "cuda": "12.4",
    "cuda_available": true
  }
}
```
