1. ​My main concern about this paper stems from an epidemic perspective, as it pertains to the model's applicability. The authors claim to produce an "assumption-free" model.  Does this imply that the model can capture the dynamics regardless of the underlying infectious disease?
Furthermore, it seems that the authors implicitly assume a "herd-immunity" strategy whereas in real world scenarios, policy makers often impose preemptive/reactive measures such as lockdowns (isolating entire subpopulations) and mandatory masks (decreasing the reproduction rate), thereby undermining the explicit "free-mixing" assumption. Since the authors haven't addressed this issue, it is unclear whether such measures would impede model learning, possibly leading to discrepancies.V

2. The terms "patient zero" and "epidemic procedure" are not properly defined prior to their use throughout the text. In addition, the term "strain" is defined on page 4, while it has already been previously used. V

3. Fig. 1 seems to lack context and motivation. I would therefore suggest to relate this Fig. to the epidemic diffusion progress (traceability) argument of section IV. B. This will also help illustrate the traceability argument. That is, in Fig. 1 the ER and BA graphs display ~3-4 distinctive regions (possibly corresponding to 4 subpopulations/strains?) evolving with respect to time, where neighboring nodes are spread along the y-axis and exhibit similar behavior (global, long-range effect), whereas for the WS and RGG graphs these regions are less apparent (disrupted) but traceability still holds locally (local, short-range effect). Do you agree with this assessment? X

4. Please comment on the observed numerical instability in the loss and spectral similarity in Fig. 4. Is this due to the variety of graph types included in the training data? X

5. The DTEF model is missing when introducing the models (on page 8, below Table I). I.e., currently only DTES, FTEF, FTES are listed. In addition, Fig. 3 should be updated as the eq. numbers appearing in it are incorrect. For example, there is no such eq. 19. V

6. The authors suggest that dimension reduction (eq. 10) as well as circumventing/avoiding random parameter initialization (eq. 15) lead to optimized results (see DTEF in Tables II-III). These methods should therefore be mentioned in contribution #3 (model efficiency) in introduction. V

7. For the sake of completeness, the authors should describe the rational for treating the strains independently within the multi-strain model, and to detail how the strains are embedded in the data. Is it possible, in principle, to have overlap between strains? Is this effect negligible?
Furthermore, the authors should also elaborate on their technique: by applying a multitude of strains, the model is fed more data, resulting in, as expected, a more comprehensive exploration of the graph, given a naive static topology. It is unclear whether this technique is indeed instrumental as the author's claim, for achieving better accuracy with more data, or whether it is simply an artifact of incremental graph coverage as portrayed in Fig . 7. X

8. In terms of writing, the results section seems quite repetitive. The authors are advised to omit, for example, Fig. 12 altogether, summarizing its results in the text as: "similarly, by varying the parameter ____ we find that...". V

9. A discussion on the model performance on real graphs is missing. V