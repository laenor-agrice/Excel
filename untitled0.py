Modela a relação linear entre variáveis.
""")
st.markdown("</div>", unsafe_allow_html=True)

with col2:
st.markdown("<div class='css-card'>", unsafe_allow_html=True)
st.markdown("### 📖 Referências Científicas")

st.markdown("""
**Livros e Artigos:**

1. Montgomery, D. C., & Runger, G. C. (2018). *Applied Statistics and Probability for Engineers*. John Wiley & Sons.

2. Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*. Academic Press.

3. von Storch, H., & Zwiers, F. W. (2019). *Statistical Analysis in Climate Research*. Cambridge University Press.

4. Hair, J. F., et al. (2019). *Multivariate Data Analysis*. Cengage Learning.

5. Helsel, D. R., & Hirsch, R. M. (2002). *Statistical Methods in Water Resources*. USGS.

**Normas Técnicas:**

- WMO (World Meteorological Organization). *Guide to Climatological Practices*. WMO-No. 100.

- INMET (Instituto Nacional de Meteorologia). *Normais Climatológicas do Brasil*.

**Software e Implementação:**

- SciPy: Virtanen, P., et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods.

- scikit-learn: Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
""")
st.markdown("</div>", unsafe_allow_html=True)

# Créditos
st.markdown("<div class='css-card' style='text-align: center;'>", unsafe_allow_html=True)
st.markdown("""
### 👨‍💻 Créditos

**MeteoAnalytics Pro v1.0**

Desenvolvido com base em metodologias científicas validadas e práticas estatísticas estabelecidas.

*Este software é fornecido para fins educacionais e de pesquisa. Os resultados devem ser validados por profissionais qualificados.*

---
© 2024 MeteoAnalytics - Todos os direitos reservados
""")
st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer
st.warning("""
⚠️ **Disclaimer Científico**: Os métodos estatísticos implementados neste software seguem as melhores práticas 
da literatura científica, porém os resultados devem ser interpretados com cautela. A qualidade dos dados de entrada 
e as premissas dos testes estatísticos devem ser verificadas antes da tomada de decisões baseadas nestas análises.
""")

if __name__ == "__main__":
main()
