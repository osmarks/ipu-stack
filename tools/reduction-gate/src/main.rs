use std::io::{self, BufRead};
use syn::punctuated::Punctuated;
use syn::{Attribute, Meta, Token, spanned::Spanned, visit::Visit};

// Unknown feature/platform predicates remain included. Only conditions proved
// false when cfg(test) is false exclude an item from production source.
fn cfg(meta: &Meta) -> Option<bool> {
    match meta {
        Meta::Path(path) if path.is_ident("test") => Some(false),
        Meta::List(list) => {
            let children = list
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .ok()?;
            let values: Vec<_> = children.iter().map(cfg).collect();
            if list.path.is_ident("not") && values.len() == 1 {
                return values[0].map(|x| !x);
            }
            if list.path.is_ident("all") {
                if values.contains(&Some(false)) {
                    Some(false)
                } else if values.iter().all(|x| *x == Some(true)) {
                    Some(true)
                } else {
                    None
                }
            } else if list.path.is_ident("any") {
                if values.contains(&Some(true)) {
                    Some(true)
                } else if values.iter().all(|x| *x == Some(false)) {
                    Some(false)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}
fn excluded(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("test")
            || (a.path().is_ident("cfg")
                && a.parse_args::<Meta>().ok().as_ref().and_then(cfg) == Some(false))
    })
}
#[derive(Default)]
struct Counter {
    excluded: Vec<(usize, usize)>,
    docs: Vec<(usize, usize)>,
    types: usize,
    variants: usize,
    fields: usize,
    functions: usize,
    macros: Vec<String>,
}
impl Counter {
    fn skip(&mut self, attrs: &[Attribute], span: proc_macro2::Span) -> bool {
        if !excluded(attrs) {
            return false;
        }
        self.excluded.push((span.start().line, span.end().line));
        true
    }
}
impl<'a> Visit<'a> for Counter {
    fn visit_attribute(&mut self, attr: &'a Attribute) {
        if attr.path().is_ident("doc") {
            self.docs
                .push((attr.span().start().line, attr.span().end().line));
        }
        syn::visit::visit_attribute(self, attr);
    }
    fn visit_item(&mut self, item: &'a syn::Item) {
        use syn::Item::*;
        let attrs: &[Attribute] = match item {
            Const(x) => &x.attrs,
            Enum(x) => &x.attrs,
            ExternCrate(x) => &x.attrs,
            Fn(x) => &x.attrs,
            ForeignMod(x) => &x.attrs,
            Impl(x) => &x.attrs,
            Macro(x) => &x.attrs,
            Mod(x) => &x.attrs,
            Static(x) => &x.attrs,
            Struct(x) => &x.attrs,
            Trait(x) => &x.attrs,
            TraitAlias(x) => &x.attrs,
            Type(x) => &x.attrs,
            Union(x) => &x.attrs,
            Use(x) => &x.attrs,
            _ => &[],
        };
        if self.skip(attrs, item.span()) {
            return;
        }
        if matches!(
            item,
            Struct(_) | Enum(_) | Union(_) | Type(_) | Trait(_) | TraitAlias(_)
        ) {
            self.types += 1;
        }
        syn::visit::visit_item(self, item);
    }
    fn visit_impl_item(&mut self, item: &'a syn::ImplItem) {
        use syn::ImplItem::*;
        let attrs: &[Attribute] = match item {
            Const(x) => &x.attrs,
            Fn(x) => &x.attrs,
            Type(x) => &x.attrs,
            Macro(x) => &x.attrs,
            _ => &[],
        };
        if self.skip(attrs, item.span()) {
            return;
        }
        if matches!(item, Type(_)) {
            self.types += 1;
        }
        syn::visit::visit_impl_item(self, item);
    }
    fn visit_trait_item(&mut self, item: &'a syn::TraitItem) {
        use syn::TraitItem::*;
        let attrs = match item {
            Const(x) => &x.attrs,
            Fn(x) => &x.attrs,
            Type(x) => &x.attrs,
            Macro(x) => &x.attrs,
            _ => return,
        };
        if self.skip(attrs, item.span()) {
            return;
        }
        if matches!(item, Type(_)) {
            self.types += 1;
        }
        syn::visit::visit_trait_item(self, item);
    }
    fn visit_foreign_item(&mut self, item: &'a syn::ForeignItem) {
        use syn::ForeignItem::*;
        let attrs = match item {
            Fn(x) => &x.attrs,
            Static(x) => &x.attrs,
            Type(x) => &x.attrs,
            Macro(x) => &x.attrs,
            _ => return,
        };
        if self.skip(attrs, item.span()) {
            return;
        }
        if matches!(item, Type(_)) {
            self.types += 1;
        }
        syn::visit::visit_foreign_item(self, item);
    }
    fn visit_signature(&mut self, item: &'a syn::Signature) {
        self.functions += 1;
        syn::visit::visit_signature(self, item);
    }
    fn visit_variant(&mut self, item: &'a syn::Variant) {
        if self.skip(&item.attrs, item.span()) {
            return;
        }
        self.variants += 1;
        syn::visit::visit_variant(self, item);
    }
    fn visit_field(&mut self, item: &'a syn::Field) {
        if self.skip(&item.attrs, item.span()) {
            return;
        }
        self.fields += 1;
        syn::visit::visit_field(self, item);
    }
    fn visit_macro(&mut self, item: &'a syn::Macro) {
        // Macro bodies are opaque to syn. Changes require human review instead
        // of pretending their generated declarations were counted.
        if !item.path.is_ident("macro_rules") && !item.path.is_ident("include") {
            return;
        }
        self.macros.push(format!(
            "{}:{}",
            item.path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
            item.tokens
        ));
    }
}
fn code_lines(tokens: proc_macro2::TokenStream, lines: &mut std::collections::BTreeSet<usize>) {
    for token in tokens {
        if let proc_macro2::TokenTree::Group(group) = token {
            lines.insert(group.span_open().start().line);
            lines.insert(group.span_close().start().line);
            code_lines(group.stream(), lines);
        } else {
            lines.extend(token.span().start().line..=token.span().end().line);
        }
    }
}
fn main() {
    for line in io::stdin().lock().lines() {
        let input: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let path = input["path"].as_str().unwrap();
        let source = input["source"].as_str().unwrap();
        let mut counter = Counter::default();
        let mut code = std::collections::BTreeSet::new();
        if path.ends_with(".rs") {
            let file = syn::parse_file(source).unwrap_or_else(|e| panic!("{path}: {e}"));
            counter.visit_file(&file);
            code_lines(source.parse().unwrap(), &mut code);
        } else {
            // Device sources use // and ordinary leading block comments. Keep
            // preprocessor lines, since these define executable contracts.
            let mut block = false;
            for (i, line) in source.lines().enumerate() {
                let mut text = line.trim();
                if block || text.starts_with("/*") {
                    if let Some(end) = text.find("*/") {
                        block = false;
                        text = text[end + 2..].trim();
                    } else {
                        block = true;
                        continue;
                    }
                }
                if !text.is_empty() && !text.starts_with("//") {
                    code.insert(i + 1);
                }
            }
        }
        let code_count = source
            .lines()
            .enumerate()
            .filter(|(i, line)| {
                code.contains(&(i + 1))
                    && !line.trim().is_empty()
                    && !counter
                        .excluded
                        .iter()
                        .chain(&counter.docs)
                        .any(|&(a, b)| a <= i + 1 && i + 1 <= b)
            })
            .count();
        println!(
            "{}",
            serde_json::json!({"lines":code_count,
            "types":counter.types, "variants":counter.variants,
            "fields":counter.fields, "functions":counter.functions,
            "macros":counter.macros})
        );
    }
}
