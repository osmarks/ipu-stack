use std::io::{self, BufRead};
use syn::{spanned::Spanned, visit::Visit};

#[derive(Default)]
struct Scan {
    excluded: Vec<(usize, usize)>,
    definitions: Vec<String>,
    imports: Vec<(String, String)>,
    paths: Vec<(usize, String)>,
}
impl Scan {
    fn imports(&mut self, tree: &syn::UseTree, prefix: String) {
        use syn::UseTree::*;
        match tree {
            Path(p) => self.imports(&p.tree, format!("{prefix}{}::", p.ident)),
            Name(n) => self
                .imports
                .push((n.ident.to_string(), format!("{prefix}{}", n.ident))),
            Rename(n) => self
                .imports
                .push((n.rename.to_string(), format!("{prefix}{}", n.ident))),
            Glob(_) => self
                .imports
                .push(("*".into(), prefix.trim_end_matches("::").into())),
            Group(g) => {
                for item in &g.items {
                    self.imports(item, prefix.clone());
                }
            }
        }
    }
}
impl<'a> Visit<'a> for Scan {
    fn visit_item(&mut self, item: &'a syn::Item) {
        if self
            .excluded
            .iter()
            .any(|&(a, b)| a <= item.span().start().line && item.span().end().line <= b)
        {
            return;
        }
        use syn::Item::*;
        let name = match item {
            Struct(x) => Some(&x.ident),
            Enum(x) => Some(&x.ident),
            Union(x) => Some(&x.ident),
            Type(x) => Some(&x.ident),
            Trait(x) => Some(&x.ident),
            Fn(x) => Some(&x.sig.ident),
            Const(x) => Some(&x.ident),
            Static(x) => Some(&x.ident),
            _ => None,
        };
        if let Some(name) = name {
            self.definitions.push(name.to_string());
        }
        // Inline modules are folded into their containing source file.
        syn::visit::visit_item(self, item);
    }
    fn visit_item_use(&mut self, item: &'a syn::ItemUse) {
        self.imports(&item.tree, String::new());
    }
    fn visit_path(&mut self, path: &'a syn::Path) {
        if !self
            .excluded
            .iter()
            .any(|&(a, b)| a <= path.span().start().line && path.span().start().line <= b)
        {
            self.paths.push((
                path.span().start().line,
                path.segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
            ));
        }
        syn::visit::visit_path(self, path);
    }
}
fn main() {
    for line in io::stdin().lock().lines() {
        let input: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let mut scan = Scan::default();
        scan.excluded = serde_json::from_value(input["excluded"].clone()).unwrap();
        scan.visit_file(&syn::parse_file(input["source"].as_str().unwrap()).unwrap());
        println!(
            "{}",
            serde_json::json!({"definitions":scan.definitions,"imports":scan.imports,"paths":scan.paths})
        );
    }
}
