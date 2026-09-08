use super::*;

impl SearchReport {
    /// Best retained candidate with no known missing capability. This is not a
    /// claim that the production beam would select it or that placement fits.
    pub fn reference(&self) -> Option<&DiagnosticMidGraph> {
        self.candidates
            .iter()
            .filter(|c| c.assumptions.is_empty())
            .min_by_key(|c| c.cycles.conservative)
    }
    /// Candidates whose optimistic score beats the retained reference. With no
    /// reference there is no justified savings claim, so this returns no hits.
    pub fn opportunities(&self) -> impl Iterator<Item = &DiagnosticMidGraph> {
        let reference = self.reference().map(|c| c.cycles.conservative);
        self.candidates.iter().filter(move |c| {
            !c.assumptions.is_empty() && reference.is_some_and(|r| c.cycles.optimistic < r)
        })
    }
}
impl DiagnosticMidGraph {
    /// Graphviz dataflow, with unsupported assumptions visible at their steps.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write;
        let mut dot = String::from(
            "digraph optimistic_mid {\n  rankdir=TB; nodesep=0.2; ranksep=0.35;\n  node [fontname=monospace,fontsize=11];\n",
        );
        for (id, value) in self.values.iter().enumerate() {
            let label = format!(
                "v{id} / high {}\n{:?}\n{:?} {:?}\n{} owners, {} replicas",
                value.origin.index(),
                value.tensor.shape.0,
                value.tensor.format.precision,
                value.tensor.format.layout.order,
                value.tensor.format.layout.tiling.tile_count,
                value.tensor.format.layout.tiling.replicas
            );
            let label = wrap_label(&label);
            writeln!(dot, "  v{id} [shape=box,label={label:?}];").unwrap();
        }
        for (id, step) in self.steps.iter().enumerate() {
            let kind = match &step.kind {
                StepKind::Algorithm { plan, .. } => format!("{:?}", plan.operator),
                StepKind::Transform(t) => format!("{:?}", t.kind),
                StepKind::FusedElementwise { operations } => format!(
                    "fused {:?}",
                    operations.iter().map(|op| &op.kind).collect::<Vec<_>>()
                ),
            };
            let label = format!(
                "{kind}\n{} / {} cycles\n{:?}",
                step.cycles.optimistic, step.cycles.conservative, step.assumptions
            );
            let label = wrap_label(&label);
            let color = if step.assumptions.is_empty() {
                "black"
            } else {
                "red"
            };
            writeln!(dot, "  s{id} [label={label:?},color={color}];").unwrap();
            for input in &step.inputs {
                writeln!(dot, "  v{input} -> s{id};").unwrap();
            }
            for output in &step.outputs {
                writeln!(dot, "  s{id} -> v{output};").unwrap();
            }
        }
        for id in &self.outputs {
            writeln!(dot, "  v{id} [penwidth=3];").unwrap();
        }
        dot.push_str("}\n");
        dot
    }
}

fn wrap_label(label: &str) -> String {
    label
        .lines()
        .map(|line| {
            let mut wrapped = String::new();
            let mut width = 0;
            for word in line.split_whitespace() {
                if width != 0 {
                    if width + 1 + word.len() > 52 {
                        wrapped.push('\n');
                        width = 0;
                    } else {
                        wrapped.push(' ');
                        width += 1;
                    }
                }
                wrapped.push_str(word);
                width += word.len();
            }
            wrapped
        })
        .collect::<Vec<_>>()
        .join("\n")
}
