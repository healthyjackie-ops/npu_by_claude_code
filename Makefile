# NPU-for-VLA simulator
SIM := python3 sim/npusim.py

.PHONY: run sweep report test help
help:
	@echo "make run     - baseline design point report"
	@echo "make sweep   - constrained design-space sweep -> Pareto"
	@echo "make report  - baseline + sensitivity curves"
	@echo "make test    - selftest (validate model vs analytic roof)"
	@echo "direct: $(SIM) run -D tensor_array_dim=192 -D dram_bw_gbps=546 --ndec 8"
run:
	@$(SIM) run
sweep:
	@$(SIM) sweep
report:
	@$(SIM) report
test:
	@$(SIM) selftest
